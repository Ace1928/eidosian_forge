from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
@add_start_docstrings('\n    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of\n    the [CLS] token) e.g. for ImageNet.\n    ', TFCVT_START_DOCSTRING)
class TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.cvt = TFCvtMainLayer(config, name='cvt')
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm')
        self.classifier = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), use_bias=True, bias_initializer='zeros', name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtForImageClassification.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""
        outputs = self.cvt(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = shape_list(sequence_output)
            sequence_output = tf.reshape(sequence_output, shape=(batch_size, num_channels, height * width))
            sequence_output = tf.transpose(sequence_output, perm=(0, 2, 1))
            sequence_output = self.layernorm(sequence_output)
        sequence_output_mean = tf.reduce_mean(sequence_output, axis=1)
        logits = self.classifier(sequence_output_mean)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'cvt', None) is not None:
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.embed_dim[-1]])
        if getattr(self, 'classifier', None) is not None:
            if hasattr(self.classifier, 'name'):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.embed_dim[-1]])