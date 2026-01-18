from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
def _check_redundant_excepthandlers(names, node):
    good = sorted(set(names), key=names.index)
    if 'BaseException' in good:
        good = ['BaseException']
    for primary, equivalents in B014.redundant_exceptions.items():
        if primary in good:
            good = [g for g in good if g not in equivalents]
    for name, other in itertools.permutations(tuple(good), 2):
        if _typesafe_issubclass(getattr(builtins, name, type), getattr(builtins, other, ())):
            if name in good:
                good.remove(name)
    if good != names:
        desc = good[0] if len(good) == 1 else '({})'.format(', '.join(good))
        as_ = ' as ' + node.name if node.name is not None else ''
        return B014(node.lineno, node.col_offset, vars=(', '.join(names), as_, desc))
    return None