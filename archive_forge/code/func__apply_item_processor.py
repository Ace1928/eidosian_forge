from __future__ import annotations
import collections.abc as collections_abc
import datetime as dt
import decimal
import enum
import json
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from uuid import UUID as _python_UUID
from . import coercions
from . import elements
from . import operators
from . import roles
from . import type_api
from .base import _NONE_NAME
from .base import NO_ARG
from .base import SchemaEventTarget
from .cache_key import HasCacheKey
from .elements import quoted_name
from .elements import Slice
from .elements import TypeCoerce as type_coerce  # noqa
from .type_api import Emulated
from .type_api import NativeForEmulated  # noqa
from .type_api import to_instance as to_instance
from .type_api import TypeDecorator as TypeDecorator
from .type_api import TypeEngine as TypeEngine
from .type_api import TypeEngineMixin
from .type_api import Variant  # noqa
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..engine import processors
from ..util import langhelpers
from ..util import OrderedDict
from ..util.typing import is_literal
from ..util.typing import Literal
from ..util.typing import typing_get_args
def _apply_item_processor(self, arr, itemproc, dim, collection_callable):
    """Helper method that can be used by bind_processor(),
        literal_processor(), etc. to apply an item processor to elements of
        an array value, taking into account the 'dimensions' for this
        array type.

        See the Postgresql ARRAY datatype for usage examples.

        .. versionadded:: 2.0

        """
    if dim is None:
        arr = list(arr)
    if dim == 1 or (dim is None and (not arr or not isinstance(arr[0], (list, tuple)))):
        if itemproc:
            return collection_callable((itemproc(x) for x in arr))
        else:
            return collection_callable(arr)
    else:
        return collection_callable((self._apply_item_processor(x, itemproc, dim - 1 if dim is not None else None, collection_callable) if x is not None else None for x in arr))