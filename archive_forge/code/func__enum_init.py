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
def _enum_init(self, enums, kw):
    """internal init for :class:`.Enum` and subclasses.

        friendly init helper used by subclasses to remove
        all the Enum-specific keyword arguments from kw.  Allows all
        other arguments in kw to pass through.

        """
    self.native_enum = kw.pop('native_enum', True)
    self.create_constraint = kw.pop('create_constraint', False)
    self.values_callable = kw.pop('values_callable', None)
    self._sort_key_function = kw.pop('sort_key_function', NO_ARG)
    length_arg = kw.pop('length', NO_ARG)
    self._omit_aliases = kw.pop('omit_aliases', True)
    _disable_warnings = kw.pop('_disable_warnings', False)
    values, objects = self._parse_into_values(enums, kw)
    self._setup_for_values(values, objects, kw)
    self.validate_strings = kw.pop('validate_strings', False)
    if self.enums:
        self._default_length = length = max((len(x) for x in self.enums))
    else:
        self._default_length = length = 0
    if length_arg is not NO_ARG:
        if not _disable_warnings and length_arg is not None and (length_arg < length):
            raise ValueError('When provided, length must be larger or equal than the length of the longest enum value. %s < %s' % (length_arg, length))
        length = length_arg
    self._valid_lookup[None] = self._object_lookup[None] = None
    super().__init__(length=length)
    if self.enum_class and values:
        kw.setdefault('name', self.enum_class.__name__.lower())
    SchemaType.__init__(self, name=kw.pop('name', None), schema=kw.pop('schema', None), metadata=kw.pop('metadata', None), inherit_schema=kw.pop('inherit_schema', False), quote=kw.pop('quote', None), _create_events=kw.pop('_create_events', True), _adapted_from=kw.pop('_adapted_from', None))