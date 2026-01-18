import collections.abc
import os
import pprint
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Sequence
from unicodedata import normalize
from _pytest import outcomes
import _pytest._code
from _pytest._io.pprint import PrettyPrinter
from _pytest._io.saferepr import saferepr
from _pytest._io.saferepr import saferepr_unlimited
from _pytest.config import Config
class _HighlightFunc(Protocol):

    def __call__(self, source: str, lexer: Literal['diff', 'python']='python') -> str:
        """Apply highlighting to the given source."""