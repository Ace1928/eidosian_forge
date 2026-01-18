import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class SpeechT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(decoder_num_layers='decoder_layers')
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(hidden_size='hidden_size', num_attention_heads='encoder_attention_heads', encoder_num_layers='encoder_layers', decoder_num_layers='decoder_layers', allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummySeq2SeqDecoderTextInputGenerator, DummySeq2SeqPastKeyValuesGenerator, DummySpeechT5InputGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator
    VARIANTS = {'with-past': 'The export follows the Transformers implementation using the KV cache, with the following components exported:\n\t - encoder_model.onnx: corresponds to the encoding part in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2544-L2556.\n\t - decoder_model.onnx: corresponds to the decoder part in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2572-L2602.\n\t - decoder_with_past_model.onnx: same as the above, with past_key_values input (KV cache filled).\n\t - decoder_postnet_and_vocoder.onnx: Decoder speech postnet and vocoder (e.g. a SpeechT5HifiGan) to generate speech from the spectrogram, as in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2605-L2614.', 'without-past': 'The same as `with-past`, just without KV cache support. This is not a recommended export as slower than `with-past`.'}
    DEFAULT_VARIANT = 'with-past'

    def __init__(self, config: 'PretrainedConfig', task: str='feature-extraction', int_dtype: str='int64', float_dtype: str='fp32', use_past: bool=False, use_past_in_inputs: bool=False, behavior: ConfigBehavior=ConfigBehavior.MONOLITH, preprocessors: Optional[List[Any]]=None, is_postnet_and_vocoder: bool=False, legacy: bool=False):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype, use_past=use_past, use_past_in_inputs=use_past_in_inputs, behavior=behavior, preprocessors=preprocessors, legacy=legacy)
        if float_dtype == 'fp16':
            raise ValueError('The ONNX export of SpeechT5 in float16 is currently not supported due to a bug in PyTorch: https://github.com/pytorch/pytorch/pull/110078. Please open an issue in Optimum if you would like to export SpeechT5 in float16.')
        self.is_postnet_and_vocoder = is_postnet_and_vocoder

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}
        if self._behavior is ConfigBehavior.ENCODER:
            common_inputs['input_ids'] = {1: 'encoder_sequence_length'}
        elif self._behavior is ConfigBehavior.DECODER:
            common_inputs['output_sequence'] = {1: 'decoder_sequence_length'}
            common_inputs['speaker_embeddings'] = {}
            common_inputs['encoder_outputs'] = {1: 'encoder_sequence_length'}
            common_inputs['encoder_attention_mask'] = {1: 'encoder_sequence_length'}
            if self.variant == 'with-past' and self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction='inputs')
        elif self.is_postnet_and_vocoder:
            common_inputs['spectrogram'] = {0: 'n_spectrums x reduction_factor'}
        else:
            raise ValueError('self._behavior is neither encoder or decoder, and is_postnet_and_vocoder=False. This should not happen.')
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {}
        if self._behavior is ConfigBehavior.ENCODER:
            common_outputs['encoder_outputs'] = {1: 'encoder_sequence_length'}
            common_outputs['encoder_attention_mask'] = {1: 'encoder_sequence_length'}
        elif self._behavior is ConfigBehavior.DECODER:
            common_outputs['output_sequence_out'] = {1: 'decoder_sequence_length + 1'}
            common_outputs['spectrum'] = {}
            common_outputs['prob'] = {}
            if self.variant == 'with-past' and self.use_past:
                self.add_past_key_values(common_outputs, direction='outputs')
        elif self.is_postnet_and_vocoder:
            common_outputs['waveform'] = {0: 'n_samples'}
        else:
            raise ValueError('self._behavior is neither encoder or decoder, and is_postnet_and_vocoder=False. This should not happen.')
        return common_outputs

    def patch_model_for_export(self, model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None) -> 'ModelPatcher':
        return SpeechT5ModelPatcher(self, model, model_kwargs=model_kwargs)

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        return {'encoder_outputs': 'encoder_hidden_states'}

    def overwrite_shape_and_generate_input(self, dummy_input_gen: 'DummyInputGenerator', input_name: str, framework: str, input_shapes: Dict):
        dummy_input_gen.batch_size = 1
        dummy_input = dummy_input_gen.generate(input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype)
        return dummy_input

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ['inputs', 'outputs']:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')
        if direction == 'inputs':
            decoder_sequence_name = 'past_decoder_sequence_length'
            name = 'past_key_values'
        else:
            decoder_sequence_name = 'past_decoder_sequence_length + 1'
            name = 'present'
        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f'{name}.{i}.decoder.key'] = {2: decoder_sequence_name}
            inputs_or_outputs[f'{name}.{i}.decoder.value'] = {2: decoder_sequence_name}
            if self.is_merged is True or (self._behavior is ConfigBehavior.DECODER and (not self.use_past_in_inputs)) or direction == 'inputs':
                inputs_or_outputs[f'{name}.{i}.encoder.key'] = {2: 'encoder_sequence_length_out'}
                inputs_or_outputs[f'{name}.{i}.encoder.value'] = {2: 'encoder_sequence_length_out'}