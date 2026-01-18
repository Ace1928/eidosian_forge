from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
from ...utils import logging
class WhisperOnnxConfig(OnnxSeq2SeqConfigWithPast):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict([('input_features', {0: 'batch', 1: 'feature_size', 2: 'encoder_sequence'})])
        if self.use_past:
            common_inputs['decoder_input_ids'] = {0: 'batch'}
        else:
            common_inputs['decoder_input_ids'] = {0: 'batch', 1: 'decoder_sequence'}
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction='inputs')
        return common_inputs

    def generate_dummy_inputs(self, preprocessor: Union['PreTrainedTokenizerBase', 'FeatureExtractionMixin'], batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None, sampling_rate: int=22050, time_duration: float=5.0, frequency: int=220) -> Mapping[str, Any]:
        dummy_inputs = OrderedDict()
        encoder_inputs = OnnxConfig.generate_dummy_inputs(self, preprocessor=preprocessor.feature_extractor, batch_size=batch_size, framework=framework, sampling_rate=sampling_rate, time_duration=time_duration, frequency=frequency)
        encoder_sequence_length = encoder_inputs['input_features'].shape[2]
        seq_length = encoder_sequence_length // 2 if self.use_past else seq_length
        decoder_inputs = super().generate_dummy_inputs(preprocessor.tokenizer, batch_size, seq_length, is_pair, framework)
        dummy_inputs['input_features'] = encoder_inputs.pop('input_features')
        dummy_inputs['decoder_input_ids'] = decoder_inputs.pop('decoder_input_ids')
        if 'past_key_values' in decoder_inputs:
            dummy_inputs['past_key_values'] = decoder_inputs.pop('past_key_values')
        return dummy_inputs

    @property
    def atol_for_validation(self) -> float:
        return 0.001