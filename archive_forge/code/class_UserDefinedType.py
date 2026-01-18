from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
class UserDefinedType(ExternalType, TypeEngineMixin, TypeEngine[_T], util.EnsureKWArg):
    """Base for user defined types.

    This should be the base of new types.  Note that
    for most cases, :class:`.TypeDecorator` is probably
    more appropriate::

      import sqlalchemy.types as types

      class MyType(types.UserDefinedType):
          cache_ok = True

          def __init__(self, precision = 8):
              self.precision = precision

          def get_col_spec(self, **kw):
              return "MYTYPE(%s)" % self.precision

          def bind_processor(self, dialect):
              def process(value):
                  return value
              return process

          def result_processor(self, dialect, coltype):
              def process(value):
                  return value
              return process

    Once the type is made, it's immediately usable::

      table = Table('foo', metadata_obj,
          Column('id', Integer, primary_key=True),
          Column('data', MyType(16))
          )

    The ``get_col_spec()`` method will in most cases receive a keyword
    argument ``type_expression`` which refers to the owning expression
    of the type as being compiled, such as a :class:`_schema.Column` or
    :func:`.cast` construct.  This keyword is only sent if the method
    accepts keyword arguments (e.g. ``**kw``) in its argument signature;
    introspection is used to check for this in order to support legacy
    forms of this function.

    The :attr:`.UserDefinedType.cache_ok` class-level flag indicates if this
    custom :class:`.UserDefinedType` is safe to be used as part of a cache key.
    This flag defaults to ``None`` which will initially generate a warning
    when the SQL compiler attempts to generate a cache key for a statement
    that uses this type.  If the :class:`.UserDefinedType` is not guaranteed
    to produce the same bind/result behavior and SQL generation
    every time, this flag should be set to ``False``; otherwise if the
    class produces the same behavior each time, it may be set to ``True``.
    See :attr:`.UserDefinedType.cache_ok` for further notes on how this works.

    .. versionadded:: 1.4.28 Generalized the :attr:`.ExternalType.cache_ok`
       flag so that it is available for both :class:`.TypeDecorator` as well
       as :class:`.UserDefinedType`.

    """
    __visit_name__ = 'user_defined'
    ensure_kwarg = 'get_col_spec'

    def coerce_compared_value(self, op: Optional[OperatorType], value: Any) -> TypeEngine[Any]:
        """Suggest a type for a 'coerced' Python value in an expression.

        Default behavior for :class:`.UserDefinedType` is the
        same as that of :class:`.TypeDecorator`; by default it returns
        ``self``, assuming the compared value should be coerced into
        the same type as this one.  See
        :meth:`.TypeDecorator.coerce_compared_value` for more detail.

        """
        return self