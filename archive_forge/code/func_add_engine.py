from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
def add_engine(self, engine, scope):
    self.add_pool(engine.pool)
    assert scope in ('class', 'global', 'function', 'fixture')
    self.testing_engines[scope].add(engine)