import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
class _Substvars_Base(typing.Generic[T], MutableMapping, ABC):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None