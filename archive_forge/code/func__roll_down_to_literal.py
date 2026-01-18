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
@classmethod
def _roll_down_to_literal(cls, element):
    is_clause_element = hasattr(element, '__clause_element__')
    if is_clause_element:
        while not isinstance(element, (elements.ClauseElement, schema.SchemaItem, type)):
            try:
                element = element.__clause_element__()
            except AttributeError:
                break
    if not is_clause_element:
        insp = inspection.inspect(element, raiseerr=False)
        if insp is not None:
            try:
                return insp.__clause_element__()
            except AttributeError:
                return insp
        return element
    else:
        return element