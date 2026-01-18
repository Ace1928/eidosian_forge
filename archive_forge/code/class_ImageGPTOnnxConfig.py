from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class ImageGPTOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'})])

    def generate_dummy_inputs(self, preprocessor: 'FeatureExtractionMixin', batch_size: int=1, seq_length: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None, num_channels: int=3, image_width: int=32, image_height: int=32) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            preprocessor ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            num_choices (`int`, *optional*, defaults to -1):
                The number of candidate answers provided for multiple choice task (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2)
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        input_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
        inputs = dict(preprocessor(images=input_image, return_tensors=framework))
        return inputs