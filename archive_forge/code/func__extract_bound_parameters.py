from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def _extract_bound_parameters(self, starting_point, result_list):
    param = object.__getattribute__(self, '_param')
    if param is not None:
        param = param._with_value(starting_point, maintain_key=True)
        result_list.append(param)
    for pywrapper in object.__getattribute__(self, '_bind_paths').values():
        getter = object.__getattribute__(pywrapper, '_getter')
        element = getter(starting_point)
        pywrapper._sa__extract_bound_parameters(element, result_list)