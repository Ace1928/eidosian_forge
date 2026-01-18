from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def _normalize_chars(name: str) -> Iterable[str]:
    for c in name:
        i = ord(c)
        if i < len(_VALID_CHARS) and _VALID_CHARS[i]:
            yield c
        else:
            yield '_'