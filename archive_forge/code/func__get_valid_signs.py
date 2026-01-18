from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def _get_valid_signs():
    signs = [False] * 128
    for i in range(len(signs)):
        c = chr(i)
        if c.isalpha() or c.isdigit() or c == '_':
            signs[i] = True
    return signs