from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def _normalize_name(name: Optional[str]) -> str:
    if name is None:
        return ''
    if validate_triad_var_name(name) and (not all((x == '_' for x in name))):
        return name
    name = name.strip()
    if name == '':
        return ''
    name = ''.join(_normalize_chars(name))
    if name[0].isdigit():
        name = '_' + name
    if validate_triad_var_name(name) and (not all((x == '_' for x in name))):
        return name
    return ''