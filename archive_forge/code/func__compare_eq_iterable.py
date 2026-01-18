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
def _compare_eq_iterable(left: Iterable[Any], right: Iterable[Any], highligher: _HighlightFunc, verbose: int=0) -> List[str]:
    if verbose <= 0 and (not running_on_ci()):
        return ['Use -v to get more diff']
    import difflib
    left_formatting = PrettyPrinter().pformat(left).splitlines()
    right_formatting = PrettyPrinter().pformat(right).splitlines()
    explanation = ['', 'Full diff:']
    explanation.extend(highligher('\n'.join((line.rstrip() for line in difflib.ndiff(right_formatting, left_formatting))), lexer='diff').splitlines())
    return explanation