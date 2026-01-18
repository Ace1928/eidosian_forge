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
def _resolve_for_python_type(self, python_type: Type[Any], matched_on: _MatchedOnType, matched_on_flattened: Type[Any]) -> Optional[Enum]:
    we_are_generic_form = self._enums_argument == [enum.Enum]
    native_enum = None
    if not we_are_generic_form and python_type is matched_on:
        enum_args = self._enums_argument
    elif is_literal(python_type):
        enum_args = typing_get_args(python_type)
        bad_args = [arg for arg in enum_args if not isinstance(arg, str)]
        if bad_args:
            raise exc.ArgumentError(f"Can't create string-based Enum datatype from non-string values: {', '.join((repr(x) for x in bad_args))}.  Please provide an explicit Enum datatype for this Python type")
        native_enum = False
    elif isinstance(python_type, type) and issubclass(python_type, enum.Enum):
        enum_args = [python_type]
    else:
        enum_args = self._enums_argument
    kw = self._make_enum_kw({})
    if native_enum is False:
        kw['native_enum'] = False
    kw['length'] = NO_ARG if self.length == 0 else self.length
    return cast(Enum, self._generic_type_affinity(_enums=enum_args, **kw))