import json
import os
import re
import unicodedata
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import regex
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
from ...utils.generic import _is_jax, _is_numpy
def _normalize(self, text: str) -> str:
    """
        Normalizes the input text. This process is for the genres and the artist

        Args:
            text (`str`):
                Artist or Genre string to normalize
        """
    accepted = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('0'), ord('9') + 1)] + ['.']
    accepted = frozenset(accepted)
    pattern = re.compile('_+')
    text = ''.join([c if c in accepted else '_' for c in text.lower()])
    text = pattern.sub('_', text).strip('_')
    return text