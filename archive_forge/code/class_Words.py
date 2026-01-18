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
class Words(str):
    lowered: str
    split_: List[str]
    first: str
    last: str

    def __init__(self, orig) -> None:
        self.lowered = self.lower()
        self.split_ = self.split()
        self.first = self.split_[0]
        self.last = self.split_[-1]