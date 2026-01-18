import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
def _parse_version_parts(s: str) -> Iterator[str]:
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)
        if not part or part == '.':
            continue
        if part[:1] in '0123456789':
            yield part.zfill(8)
        else:
            yield ('*' + part)
    yield '*final'