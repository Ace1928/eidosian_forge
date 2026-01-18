import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
def _legacy_cmpkey(version: str) -> LegacyCmpKey:
    epoch = -1
    parts: List[str] = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith('*'):
            if part < '*final':
                while parts and parts[-1] == '*final-':
                    parts.pop()
            while parts and parts[-1] == '00000000':
                parts.pop()
        parts.append(part)
    return (epoch, tuple(parts))