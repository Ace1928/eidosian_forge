from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Optional
from transformers.file_utils import TensorType
from transformers.utils import logging
from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
class EncoderOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('attention_mask', {0: 'batch', 1: 'sequence'})])

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return OrderedDict({'last_hidden_state': {0: 'batch', 1: 'sequence'}})