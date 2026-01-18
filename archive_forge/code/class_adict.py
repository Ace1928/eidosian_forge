from __future__ import annotations
from collections import deque
import decimal
import gc
from itertools import chain
import random
import sys
from sys import getsizeof
import types
from . import config
from . import mock
from .. import inspect
from ..engine import Connection
from ..schema import Column
from ..schema import DropConstraint
from ..schema import DropTable
from ..schema import ForeignKeyConstraint
from ..schema import MetaData
from ..schema import Table
from ..sql import schema
from ..sql.sqltypes import Integer
from ..util import decorator
from ..util import defaultdict
from ..util import has_refcount_gc
from ..util import inspect_getfullargspec
class adict(dict):
    """Dict keys available as attributes.  Shadows."""

    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            return dict.__getattribute__(self, key)

    def __call__(self, *keys):
        return tuple([self[key] for key in keys])
    get_all = __call__