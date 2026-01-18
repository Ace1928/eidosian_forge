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
class _WordMeta(type):

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, str) and len(instance) >= 1