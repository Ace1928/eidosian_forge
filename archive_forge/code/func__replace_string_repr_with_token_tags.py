import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def _replace_string_repr_with_token_tags(prompt: str) -> str:
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt