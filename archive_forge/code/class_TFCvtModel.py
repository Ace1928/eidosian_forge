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
@add_start_docstrings('The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.', TFCVT_START_DOCSTRING)
class TFCvtModel(TFCvtPreTrainedModel):

    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.cvt = TFCvtMainLayer(config, name='cvt')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithCLSToken, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtModel.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        outputs = self.cvt(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return (outputs[0],) + outputs[1:]
        return TFBaseModelOutputWithCLSToken(last_hidden_state=outputs.last_hidden_state, cls_token_value=outputs.cls_token_value, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'cvt', None) is not None:
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)