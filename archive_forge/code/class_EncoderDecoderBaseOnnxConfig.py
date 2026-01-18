from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
class EncoderDecoderBaseOnnxConfig(OnnxSeq2SeqConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    def __init__(self, config: 'PretrainedConfig', task: str='feature-extraction', int_dtype: str='int64', float_dtype: str='fp32', use_past: bool=False, use_past_in_inputs: bool=False, behavior: ConfigBehavior=ConfigBehavior.MONOLITH, preprocessors: Optional[List[Any]]=None, legacy: bool=False):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype, use_past=use_past, use_past_in_inputs=use_past_in_inputs, behavior=behavior, preprocessors=preprocessors, legacy=legacy)
        from ..tasks import TasksManager
        self.is_decoder_with_past = False
        encoder_onnx_config_constructor = TasksManager.get_exporter_config_constructor(exporter='onnx', task='feature-extraction', model_type=config.encoder.model_type, library_name='transformers')
        self._encoder_onnx_config = encoder_onnx_config_constructor(config.encoder, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors)
        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = self._encoder_onnx_config._normalized_config
        decoder_onnx_config_constructor = TasksManager.get_exporter_config_constructor(exporter='onnx', task='feature-extraction', model_type=config.decoder.model_type, library_name='transformers')
        kwargs = {}
        if issubclass(decoder_onnx_config_constructor.func, OnnxConfigWithPast):
            self.is_decoder_with_past = True
            kwargs['use_past'] = use_past
        else:
            self.use_past = False
        if use_past and (not self.is_decoder_with_past):
            raise ValueError(f'The decoder part of the encoder-decoder model is {config.decoder.model_type} which does not need past key values.')
        self._decoder_onnx_config = decoder_onnx_config_constructor(config.decoder, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors, **kwargs)
        if issubclass(decoder_onnx_config_constructor.func, OnnxSeq2SeqConfigWithPast):
            self._decoder_onnx_config = self._decoder_onnx_config.with_behavior(self._behavior, use_past=kwargs['use_past'], use_past_in_inputs=use_past_in_inputs)
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS = self._decoder_onnx_config._normalized_config
        if isinstance(self._decoder_onnx_config, OnnxSeq2SeqConfigWithPast):
            self._past_key_values_generator = (DummySeq2SeqDecoderTextInputGenerator, DummySeq2SeqPastKeyValuesGenerator)
        else:
            self._past_key_values_generator = (DummySeq2SeqDecoderTextInputGenerator, DummyPastKeyValuesGenerator)
        self.DUMMY_INPUT_GENERATOR_CLASSES += self._past_key_values_generator

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs['input_ids'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        common_inputs['attention_mask'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        if self._behavior is not ConfigBehavior.ENCODER:
            common_inputs.pop('attention_mask')
            if self.use_past_in_inputs:
                common_inputs['decoder_input_ids'] = {0: 'batch_size'}
            else:
                common_inputs['decoder_input_ids'] = {0: 'batch_size', 1: 'decoder_sequence_length'}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction='inputs')
        if self._behavior is ConfigBehavior.DECODER:
            common_inputs['encoder_outputs'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        return common_inputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {'decoder_input_ids': 'input_ids', 'encoder_outputs': 'encoder_hidden_states', 'attention_mask': 'encoder_attention_mask'}
        return {}

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if self.is_decoder_with_past:
            return self._decoder_onnx_config.add_past_key_values(inputs_or_outputs, direction)

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if self.is_decoder_with_past:
            return self._decoder_onnx_config.flatten_past_key_values(flattened_output, name, idx, t)

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        return self._decoder_onnx_config.flatten_output_collection_property(name, field)

    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]]=None) -> Dict[str, Any]:
        if self._behavior is ConfigBehavior.ENCODER:
            return self._encoder_onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)
        else:
            if self._behavior is ConfigBehavior.DECODER:
                reference_model_inputs['input_ids'] = reference_model_inputs.pop('decoder_input_ids')
            if 'encoder_outputs' in reference_model_inputs:
                if 'encoder_hidden_states' in onnx_input_names:
                    reference_model_inputs['encoder_hidden_states'] = reference_model_inputs.pop('encoder_outputs')[0]
                else:
                    reference_model_inputs.pop('encoder_outputs')
            return self._decoder_onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)

    def post_process_exported_models(self, path: Path, models_and_onnx_configs: Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], 'OnnxConfig']], onnx_files_subpaths: List[str]):
        models_and_onnx_configs, onnx_files_subpaths = super().post_process_exported_models(path, models_and_onnx_configs, onnx_files_subpaths)
        if self.use_past is True and len(models_and_onnx_configs) == 3:
            models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.is_merged = True
            models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.use_cache_branch = False
            models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.use_past_in_inputs = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1]._decoder_onnx_config.use_cache_branch = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1]._decoder_onnx_config.is_merged = True
        return (models_and_onnx_configs, onnx_files_subpaths)