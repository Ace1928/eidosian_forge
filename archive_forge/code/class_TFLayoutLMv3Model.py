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
@add_start_docstrings('The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.', LAYOUTLMV3_START_DOCSTRING)
class TFLayoutLMv3Model(TFLayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['position_ids']

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name='layoutlmv3')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: tf.Tensor | None=None, bbox: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModel
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModel.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="tf")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        outputs = self.layoutlmv3(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layoutlmv3', None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)