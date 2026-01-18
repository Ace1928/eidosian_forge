import collections
import itertools
import json
import os
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import add_end_docstrings, logging
def _convert_token_to_shape_id(self, token):
    """Converts a token (str) in an shape_id using the shape vocab."""
    return self.word_shape.get(token, self.word_shape.get(self.unk_token))