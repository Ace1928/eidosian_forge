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
def _indef_article(self, word: str, count: Union[int, str, Any]) -> str:
    mycount = self.get_count(count)
    if mycount != 1:
        return f'{count} {word}'
    value = self.ud_match(word, self.A_a_user_defined)
    if value is not None:
        return f'{value} {word}'
    matches = (f'{article} {word}' for regexen, article in self._indef_article_cases if regexen.search(word))
    fallback = f'a {word}'
    return next(matches, fallback)