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
def _make_enum_kw(self, kw):
    kw.setdefault('validate_strings', self.validate_strings)
    if self.name:
        kw.setdefault('name', self.name)
    kw.setdefault('schema', self.schema)
    kw.setdefault('inherit_schema', self.inherit_schema)
    kw.setdefault('metadata', self.metadata)
    kw.setdefault('native_enum', self.native_enum)
    kw.setdefault('values_callable', self.values_callable)
    kw.setdefault('create_constraint', self.create_constraint)
    kw.setdefault('length', self.length)
    kw.setdefault('omit_aliases', self._omit_aliases)
    return kw