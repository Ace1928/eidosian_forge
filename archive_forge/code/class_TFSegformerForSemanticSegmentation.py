from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
@add_start_docstrings('SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.', SEGFORMER_START_DOCSTRING)
class TFSegformerForSemanticSegmentation(TFSegformerPreTrainedModel):

    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.segformer = TFSegformerMainLayer(config, name='segformer')
        self.decode_head = TFSegformerDecodeHead(config, name='decode_head')

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
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TFSemanticSegmenterOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a (per-pixel) classification loss is computed
            (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs, training=False)
        >>> # logits are of shape (batch_size, num_labels, height/4, width/4)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.segformer(pixel_values, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        logits = self.decode_head(encoder_hidden_states)
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
        return TFSemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'segformer', None) is not None:
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
        if getattr(self, 'decode_head', None) is not None:
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)