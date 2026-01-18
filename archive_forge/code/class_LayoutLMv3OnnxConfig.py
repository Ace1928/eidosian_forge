from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import logging
class LayoutLMv3OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse('1.12')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task in ['question-answering', 'sequence-classification']:
            return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('attention_mask', {0: 'batch', 1: 'sequence'}), ('bbox', {0: 'batch', 1: 'sequence'}), ('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'})])
        else:
            return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('bbox', {0: 'batch', 1: 'sequence'}), ('attention_mask', {0: 'batch', 1: 'sequence'}), ('pixel_values', {0: 'batch', 1: 'num_channels'})])

    @property
    def atol_for_validation(self) -> float:
        return 1e-05

    @property
    def default_onnx_opset(self) -> int:
        return 12

    def generate_dummy_inputs(self, processor: 'ProcessorMixin', batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None, num_channels: int=3, image_width: int=40, image_height: int=40) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            processor ([`ProcessorMixin`]):
                The processor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2).
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the processor will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Any]: holding the kwargs to provide to the model's forward function
        """
        setattr(processor.image_processor, 'apply_ocr', False)
        batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0)
        token_to_add = processor.tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add)
        dummy_text = [[' '.join([processor.tokenizer.unk_token]) * seq_length]] * batch_size
        dummy_bboxes = [[[48, 84, 73, 128]]] * batch_size
        dummy_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
        inputs = dict(processor(dummy_image, text=dummy_text, boxes=dummy_bboxes, return_tensors=framework))
        return inputs