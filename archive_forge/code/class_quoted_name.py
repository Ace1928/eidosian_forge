from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class quoted_name(util.MemoizedSlots, str):
    """Represent a SQL identifier combined with quoting preferences.

    :class:`.quoted_name` is a Python unicode/str subclass which
    represents a particular identifier name along with a
    ``quote`` flag.  This ``quote`` flag, when set to
    ``True`` or ``False``, overrides automatic quoting behavior
    for this identifier in order to either unconditionally quote
    or to not quote the name.  If left at its default of ``None``,
    quoting behavior is applied to the identifier on a per-backend basis
    based on an examination of the token itself.

    A :class:`.quoted_name` object with ``quote=True`` is also
    prevented from being modified in the case of a so-called
    "name normalize" option.  Certain database backends, such as
    Oracle, Firebird, and DB2 "normalize" case-insensitive names
    as uppercase.  The SQLAlchemy dialects for these backends
    convert from SQLAlchemy's lower-case-means-insensitive convention
    to the upper-case-means-insensitive conventions of those backends.
    The ``quote=True`` flag here will prevent this conversion from occurring
    to support an identifier that's quoted as all lower case against
    such a backend.

    The :class:`.quoted_name` object is normally created automatically
    when specifying the name for key schema constructs such as
    :class:`_schema.Table`, :class:`_schema.Column`, and others.
    The class can also be
    passed explicitly as the name to any function that receives a name which
    can be quoted.  Such as to use the :meth:`_engine.Engine.has_table`
    method with
    an unconditionally quoted name::

        from sqlalchemy import create_engine
        from sqlalchemy import inspect
        from sqlalchemy.sql import quoted_name

        engine = create_engine("oracle+cx_oracle://some_dsn")
        print(inspect(engine).has_table(quoted_name("some_table", True)))

    The above logic will run the "has table" logic against the Oracle backend,
    passing the name exactly as ``"some_table"`` without converting to
    upper case.

    .. versionchanged:: 1.2 The :class:`.quoted_name` construct is now
       importable from ``sqlalchemy.sql``, in addition to the previous
       location of ``sqlalchemy.sql.elements``.

    """
    __slots__ = ('quote', 'lower', 'upper')
    quote: Optional[bool]

    @overload
    @classmethod
    def construct(cls, value: str, quote: Optional[bool]) -> quoted_name:
        ...

    @overload
    @classmethod
    def construct(cls, value: None, quote: Optional[bool]) -> None:
        ...

    @classmethod
    def construct(cls, value: Optional[str], quote: Optional[bool]) -> Optional[quoted_name]:
        if value is None:
            return None
        else:
            return quoted_name(value, quote)

    def __new__(cls, value: str, quote: Optional[bool]) -> quoted_name:
        assert value is not None, 'use quoted_name.construct() for None passthrough'
        if isinstance(value, cls) and (quote is None or value.quote == quote):
            return value
        self = super().__new__(cls, value)
        self.quote = quote
        return self

    def __reduce__(self):
        return (quoted_name, (str(self), self.quote))

    def _memoized_method_lower(self):
        if self.quote:
            return self
        else:
            return str(self).lower()

    def _memoized_method_upper(self):
        if self.quote:
            return self
        else:
            return str(self).upper()