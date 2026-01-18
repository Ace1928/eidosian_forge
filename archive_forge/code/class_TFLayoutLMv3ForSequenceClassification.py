from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
@add_start_docstrings('\n    LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the\n    [CLS] token) e.g. for document image classification tasks such as the\n    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.\n    ', LAYOUTLMV3_START_DOCSTRING)
class TFLayoutLMv3ForSequenceClassification(TFLayoutLMv3PreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['position_ids']

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name='layoutlmv3')
        self.classifier = TFLayoutLMv3ClassificationHead(config, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, bbox: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForSequenceClassification
        >>> from datasets import load_dataset
        >>> import tensorflow as tf

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="tf")
        >>> sequence_label = tf.convert_to_tensor([1])

        >>> outputs = model(**encoding, labels=sequence_label)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.layoutlmv3(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, bbox=bbox, pixel_values=pixel_values, training=training)
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layoutlmv3', None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)