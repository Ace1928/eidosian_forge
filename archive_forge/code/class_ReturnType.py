import enum
import warnings
from typing import Dict
from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import Pipeline, build_pipeline_init_args
class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2