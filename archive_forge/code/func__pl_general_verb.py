import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def _pl_general_verb(self, word: str, count: Optional[Union[str, int]]=None) -> str:
    count = self.get_count(count)
    if count == 1:
        return word
    mo = plverb_ambiguous_pres_keys.search(word)
    if mo:
        return f'{plverb_ambiguous_pres[mo.group(1).lower()]}{mo.group(2)}'
    mo = plverb_ambiguous_non_pres.search(word)
    if mo:
        return word
    return word