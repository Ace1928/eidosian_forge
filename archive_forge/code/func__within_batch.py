from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
@contextlib.contextmanager
def _within_batch(self) -> Iterator[None]:
    self._has_batch = True
    yield
    self._has_batch = False