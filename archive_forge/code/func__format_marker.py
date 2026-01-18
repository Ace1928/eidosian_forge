import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ._parser import (
from ._parser import (
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name
def _format_marker(marker: Union[List[str], MarkerAtom, str], first: Optional[bool]=True) -> str:
    assert isinstance(marker, (list, tuple, str))
    if isinstance(marker, list) and len(marker) == 1 and isinstance(marker[0], (list, tuple)):
        return _format_marker(marker[0])
    if isinstance(marker, list):
        inner = (_format_marker(m, first=False) for m in marker)
        if first:
            return ' '.join(inner)
        else:
            return '(' + ' '.join(inner) + ')'
    elif isinstance(marker, tuple):
        return ' '.join([m.serialize() for m in marker])
    else:
        return marker