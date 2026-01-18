import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
@staticmethod
def _compute_offsets(char_repetitions: List[int], chars: List[str], ctc_token: int, word_delimiter_token: Optional[int]=None) -> List[Dict[str, Union[str, int]]]:
    end_indices = np.asarray(char_repetitions).cumsum()
    start_indices = np.concatenate(([0], end_indices[:-1]))
    offsets = [{'char': t, 'start_offset': s, 'end_offset': e} for t, s, e in zip(chars, start_indices, end_indices)]
    offsets = list(filter(lambda offsets: offsets['char'] != ctc_token, offsets))
    if word_delimiter_token is not None:
        offsets = list(filter(lambda offsets: offsets['char'] != word_delimiter_token, offsets))
    return offsets