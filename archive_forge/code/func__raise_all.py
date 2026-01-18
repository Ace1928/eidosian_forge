from __future__ import annotations
from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import contextlib
from dataclasses import dataclass, field
import functools
from .compat import io
import itertools
import os
import re
import sys
from typing import Iterable
@staticmethod
def _raise_all(exceptions: Iterable['ParsingError']):
    """
        Combine any number of ParsingErrors into one and raise it.
        """
    exceptions = iter(exceptions)
    with contextlib.suppress(StopIteration):
        raise next(exceptions).combine(exceptions)