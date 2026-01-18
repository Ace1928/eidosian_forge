import difflib
from bisect import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple
def _check_consistency(answer):
    next_a = -1
    next_b = -1
    for a, b, match_len in answer:
        if a < next_a:
            raise ValueError('Non increasing matches for a')
        if b < next_b:
            raise ValueError('Non increasing matches for b')
        next_a = a + match_len
        next_b = b + match_len