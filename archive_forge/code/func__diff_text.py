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
def _diff_text(left: str, right: str, verbose: int=0) -> List[str]:
    """Return the explanation for the diff between text.

    Unless --verbose is used this will skip leading and trailing
    characters which are identical to keep the diff minimal.
    """
    from difflib import ndiff
    explanation: List[str] = []
    if verbose < 1:
        i = 0
        for i in range(min(len(left), len(right))):
            if left[i] != right[i]:
                break
        if i > 42:
            i -= 10
            explanation = ['Skipping %s identical leading characters in diff, use -v to show' % i]
            left = left[i:]
            right = right[i:]
        if len(left) == len(right):
            for i in range(len(left)):
                if left[-i] != right[-i]:
                    break
            if i > 42:
                i -= 10
                explanation += [f'Skipping {i} identical trailing characters in diff, use -v to show']
                left = left[:-i]
                right = right[:-i]
    keepends = True
    if left.isspace() or right.isspace():
        left = repr(str(left))
        right = repr(str(right))
        explanation += ['Strings contain only whitespace, escaping them using repr()']
    explanation += [line.strip('\n') for line in ndiff(right.splitlines(keepends), left.splitlines(keepends))]
    return explanation