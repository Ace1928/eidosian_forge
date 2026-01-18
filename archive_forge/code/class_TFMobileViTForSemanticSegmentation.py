from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
@add_start_docstrings('\n    MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.\n    ', MOBILEVIT_START_DOCSTRING)
class TFMobileViTForSemanticSegmentation(TFMobileViTPreTrainedModel):

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.mobilevit = TFMobileViTMainLayer(config, expand_output=False, name='mobilevit')
        self.segmentation_head = TFMobileViTDeepLabV3(config, name='segmentation_head')

    def hf_compute_loss(self, logits, labels):
        label_interp_shape = shape_list(labels)[1:]
        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method='bilinear')
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        def masked_loss(real, pred):
            unmasked_loss = loss_fct(real, pred)
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))
        return masked_loss(labels, upsampled_logits)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFSemanticSegmenterOutputWithNoAttention]:
        """
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
        >>> model = TFMobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

        >>> inputs = image_processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mobilevit(pixel_values, output_hidden_states=True, return_dict=return_dict, training=training)
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        logits = self.segmentation_head(encoder_hidden_states, training=training)
        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError('The number of labels should be greater than one')
            else:
                loss = self.hf_compute_loss(logits=logits, labels=labels)
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSemanticSegmenterOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'mobilevit', None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        if getattr(self, 'segmentation_head', None) is not None:
            with tf.name_scope(self.segmentation_head.name):
                self.segmentation_head.build(None)