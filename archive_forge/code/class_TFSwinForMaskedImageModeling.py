from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
@add_start_docstrings('Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).', SWIN_START_DOCSTRING)
class TFSwinForMaskedImageModeling(TFSwinPreTrainedModel):

    def __init__(self, config: SwinConfig):
        super().__init__(config)
        self.swin = TFSwinMainLayer(config, add_pooling_layer=False, use_mask_token=True, name='swin')
        self.decoder = TFSwinDecoder(config, name='decoder')

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSwinMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFSwinMaskedImageModelingOutput]:
        """
        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, TFSwinForMaskedImageModeling
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        >>> model = TFSwinForMaskedImageModeling.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="tf").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = tf.random.uniform((1, num_patches)) >= 0.5

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swin(pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = tf.transpose(sequence_output, (0, 2, 1))
        batch_size, num_channels, sequence_length = shape_list(sequence_output)
        height = width = int(sequence_length ** 0.5)
        sequence_output = tf.reshape(sequence_output, (batch_size, num_channels, height, width))
        reconstructed_pixel_values = self.decoder(sequence_output)
        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = tf.reshape(bool_masked_pos, (-1, size, size))
            mask = tf.repeat(bool_masked_pos, self.config.patch_size, 1)
            mask = tf.repeat(mask, self.config.patch_size, 2)
            mask = tf.expand_dims(mask, 1)
            mask = tf.cast(mask, tf.float32)
            reconstruction_loss = keras.losses.mean_absolute_error(tf.transpose(pixel_values, (1, 2, 3, 0)), tf.transpose(reconstructed_pixel_values, (1, 2, 3, 0)))
            reconstruction_loss = tf.expand_dims(reconstruction_loss, 0)
            total_loss = tf.reduce_sum(reconstruction_loss * mask)
            num_masked_pixels = (tf.reduce_sum(mask) + 1e-05) * self.config.num_channels
            masked_im_loss = total_loss / num_masked_pixels
            masked_im_loss = tf.reshape(masked_im_loss, (1,))
        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[2:]
            return (masked_im_loss,) + output if masked_im_loss is not None else output
        return TFSwinMaskedImageModelingOutput(loss=masked_im_loss, reconstruction=reconstructed_pixel_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, reshaped_hidden_states=outputs.reshaped_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'swin', None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)
        if getattr(self, 'decoder', None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)