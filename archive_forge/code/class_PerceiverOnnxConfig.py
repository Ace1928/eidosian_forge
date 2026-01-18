from collections import OrderedDict
from typing import Any, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import FeatureExtractionMixin
from ...onnx import OnnxConfig
from ...onnx.utils import compute_effective_axis_dimension
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import TensorType, logging
class PerceiverOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == 'multiple-choice':
            dynamic_axis = {0: 'batch', 1: 'choice', 2: 'sequence'}
        else:
            dynamic_axis = {0: 'batch', 1: 'sequence'}
        return OrderedDict([('inputs', dynamic_axis), ('attention_mask', dynamic_axis)])

    @property
    def atol_for_validation(self) -> float:
        return 0.0001

    def generate_dummy_inputs(self, preprocessor: Union['PreTrainedTokenizerBase', 'FeatureExtractionMixin'], batch_size: int=-1, seq_length: int=-1, num_choices: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None, num_channels: int=3, image_width: int=40, image_height: int=40) -> Mapping[str, Any]:
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0)
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add)
            dummy_input = [' '.join(['a']) * seq_length] * batch_size
            inputs = dict(preprocessor(dummy_input, return_tensors=framework))
            inputs['inputs'] = inputs.pop('input_ids')
            return inputs
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == 'pixel_values':
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            inputs = dict(preprocessor(images=dummy_input, return_tensors=framework))
            inputs['inputs'] = inputs.pop('pixel_values')
            return inputs
        else:
            raise ValueError('Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor.')