import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class MatchFunc(typing.Protocol):

    def __call__(self, string: str, pos: int=..., endpos: int=...) -> Optional['re.Match[str]']:
        ...