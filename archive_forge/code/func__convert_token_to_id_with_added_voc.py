import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, overload
from .tokenization_utils_base import (
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging
def _convert_token_to_id_with_added_voc(self, token):
    if token is None:
        return None
    if token in self._added_tokens_encoder:
        return self._added_tokens_encoder[token]
    return self._convert_token_to_id(token)