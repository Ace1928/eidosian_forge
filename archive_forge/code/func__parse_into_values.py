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
def _parse_into_values(self, enums, kw):
    if not enums and '_enums' in kw:
        enums = kw.pop('_enums')
    if len(enums) == 1 and hasattr(enums[0], '__members__'):
        self.enum_class = enums[0]
        _members = self.enum_class.__members__
        if self._omit_aliases is True:
            members = OrderedDict(((n, v) for n, v in _members.items() if v.name == n))
        else:
            members = _members
        if self.values_callable:
            values = self.values_callable(self.enum_class)
        else:
            values = list(members)
        objects = [members[k] for k in members]
        return (values, objects)
    else:
        self.enum_class = None
        return (enums, enums)