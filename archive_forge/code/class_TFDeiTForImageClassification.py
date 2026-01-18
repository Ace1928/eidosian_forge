from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_deit import DeiTConfig
@add_start_docstrings('\n    DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of\n    the [CLS] token) e.g. for ImageNet.\n    ', DEIT_START_DOCSTRING)
class TFDeiTForImageClassification(TFDeiTPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: DeiTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name='deit')
        self.classifier = keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tf.Tensor, TFImageClassifierOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFDeiTForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> keras.utils.set_random_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # note: we are loading a TFDeiTForImageClassificationWithTeacher from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        >>> model = TFDeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        Predicted class: little blue heron, Egretta caerulea
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'deit', None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])