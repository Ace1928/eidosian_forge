from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
class TextSeq2SeqOnnxConfig(OnnxSeq2SeqConfigWithPast):
    """
    Handles encoder-decoder-based text architectures.
    """
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummySeq2SeqDecoderTextInputGenerator, DummySeq2SeqPastKeyValuesGenerator)

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {'decoder_input_ids': 'input_ids', 'encoder_outputs': 'encoder_hidden_states', 'attention_mask': 'encoder_attention_mask'}
        return {}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs['input_ids'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        common_inputs['attention_mask'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs['decoder_input_ids'] = {0: 'batch_size'}
                self.add_past_key_values(common_inputs, direction='inputs')
            else:
                common_inputs['decoder_input_ids'] = {0: 'batch_size', 1: 'decoder_sequence_length'}
        if self._behavior is ConfigBehavior.DECODER:
            common_inputs['encoder_outputs'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        return common_inputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> List['DummyInputGenerator']:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config, **kwargs)
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](self.task, self._normalized_config, **kwargs)
        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2](self.task, self._normalized_config, encoder_sequence_length=dummy_text_input_generator.sequence_length, **kwargs)
        dummy_inputs_generators = [dummy_text_input_generator, dummy_decoder_text_input_generator, dummy_seq2seq_past_key_values_generator]
        return dummy_inputs_generators