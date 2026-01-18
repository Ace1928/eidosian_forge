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
def _pl_special_adjective(self, word: str, count: Optional[Union[str, int]]=None) -> Union[str, bool]:
    count = self.get_count(count)
    if count == 1:
        return word
    value = self.ud_match(word, self.pl_adj_user_defined)
    if value is not None:
        return value
    mo = pl_adj_special_keys.search(word)
    if mo:
        return pl_adj_special[mo.group(1).lower()]
    mo = pl_adj_poss_keys.search(word)
    if mo:
        return pl_adj_poss[mo.group(1).lower()]
    mo = ENDS_WITH_APOSTROPHE_S.search(word)
    if mo:
        pl = self.plural_noun(mo.group(1))
        trailing_s = '' if pl[-1] == 's' else 's'
        return f"{pl}'{trailing_s}"
    return False