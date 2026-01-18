import importlib.metadata
import itertools
import string
from typing import Dict, List, Tuple
def _tuple_from_text(version: str) -> Tuple:
    text_parts = version.split('.')
    int_parts = []
    for text_part in text_parts:
        digit_prefix = ''.join(itertools.takewhile(lambda x: x in string.digits, text_part))
        try:
            int_parts.append(int(digit_prefix))
        except Exception:
            break
    return tuple(int_parts)