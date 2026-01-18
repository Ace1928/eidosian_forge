import copy
import enum
import gc
import inspect
import itertools
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.utils import is_accelerate_available, is_torch_available
from ...onnx import remove_duplicate_weights_from_tied_info
from ...onnx import merge_decoders
from ...utils import (
from ...utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ...utils import TRANSFORMERS_MINIMUM_VERSION as GLOBAL_MIN_TRANSFORMERS_VERSION
from ...utils.doc import add_dynamic_docstring
from ...utils.import_utils import check_if_transformers_greater, is_onnx_available, is_onnxruntime_available
from ..base import ExportConfig
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import ModelPatcher, Seq2SeqModelPatcher
class ConfigBehavior(str, enum.Enum):
    """
    Specifies the behavior of the [`~exporters.onnx.base.OnnxSeq2SeqConfigWithPast`]:
        - MONOLITH: the config can be used to export the whole seq2seq model as a single file.
        - ENCODER: the config can be used to export the encoder part of the seq2seq model.
        - DECODER: the config can be used to export the decoder part of the seq2seq model.
    """
    MONOLITH = 'monolith'
    ENCODER = 'encoder'
    DECODER = 'decoder'