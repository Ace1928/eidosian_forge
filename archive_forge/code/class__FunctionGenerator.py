from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
class _FunctionGenerator:
    """Generate SQL function expressions.

    :data:`.func` is a special object instance which generates SQL
    functions based on name-based attributes, e.g.:

    .. sourcecode:: pycon+sql

        >>> print(func.count(1))
        {printsql}count(:param_1)

    The returned object is an instance of :class:`.Function`, and  is a
    column-oriented SQL element like any other, and is used in that way:

    .. sourcecode:: pycon+sql

        >>> print(select(func.count(table.c.id)))
        {printsql}SELECT count(sometable.id) FROM sometable

    Any name can be given to :data:`.func`. If the function name is unknown to
    SQLAlchemy, it will be rendered exactly as is. For common SQL functions
    which SQLAlchemy is aware of, the name may be interpreted as a *generic
    function* which will be compiled appropriately to the target database:

    .. sourcecode:: pycon+sql

        >>> print(func.current_timestamp())
        {printsql}CURRENT_TIMESTAMP

    To call functions which are present in dot-separated packages,
    specify them in the same manner:

    .. sourcecode:: pycon+sql

        >>> print(func.stats.yield_curve(5, 10))
        {printsql}stats.yield_curve(:yield_curve_1, :yield_curve_2)

    SQLAlchemy can be made aware of the return type of functions to enable
    type-specific lexical and result-based behavior. For example, to ensure
    that a string-based function returns a Unicode value and is similarly
    treated as a string in expressions, specify
    :class:`~sqlalchemy.types.Unicode` as the type:

    .. sourcecode:: pycon+sql

        >>> print(func.my_string(u'hi', type_=Unicode) + ' ' +
        ...       func.my_string(u'there', type_=Unicode))
        {printsql}my_string(:my_string_1) || :my_string_2 || my_string(:my_string_3)

    The object returned by a :data:`.func` call is usually an instance of
    :class:`.Function`.
    This object meets the "column" interface, including comparison and labeling
    functions.  The object can also be passed the :meth:`~.Connectable.execute`
    method of a :class:`_engine.Connection` or :class:`_engine.Engine`,
    where it will be
    wrapped inside of a SELECT statement first::

        print(connection.execute(func.current_timestamp()).scalar())

    In a few exception cases, the :data:`.func` accessor
    will redirect a name to a built-in expression such as :func:`.cast`
    or :func:`.extract`, as these names have well-known meaning
    but are not exactly the same as "functions" from a SQLAlchemy
    perspective.

    Functions which are interpreted as "generic" functions know how to
    calculate their return type automatically. For a listing of known generic
    functions, see :ref:`generic_functions`.

    .. note::

        The :data:`.func` construct has only limited support for calling
        standalone "stored procedures", especially those with special
        parameterization concerns.

        See the section :ref:`stored_procedures` for details on how to use
        the DBAPI-level ``callproc()`` method for fully traditional stored
        procedures.

    .. seealso::

        :ref:`tutorial_functions` - in the :ref:`unified_tutorial`

        :class:`.Function`

    """

    def __init__(self, **opts: Any):
        self.__names: List[str] = []
        self.opts = opts

    def __getattr__(self, name: str) -> _FunctionGenerator:
        if name.startswith('__'):
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)
        elif name.endswith('_'):
            name = name[0:-1]
        f = _FunctionGenerator(**self.opts)
        f.__names = list(self.__names) + [name]
        return f

    @overload
    def __call__(self, *c: Any, type_: _TypeEngineArgument[_T], **kwargs: Any) -> Function[_T]:
        ...

    @overload
    def __call__(self, *c: Any, **kwargs: Any) -> Function[Any]:
        ...

    def __call__(self, *c: Any, **kwargs: Any) -> Function[Any]:
        o = self.opts.copy()
        o.update(kwargs)
        tokens = len(self.__names)
        if tokens == 2:
            package, fname = self.__names
        elif tokens == 1:
            package, fname = ('_default', self.__names[0])
        else:
            package = None
        if package is not None:
            func = _registry[package].get(fname.lower())
            if func is not None:
                return func(*c, **o)
        return Function(self.__names[-1], *c, packagenames=tuple(self.__names[0:-1]), **o)
    if TYPE_CHECKING:

        @property
        def aggregate_strings(self) -> Type[aggregate_strings]:
            ...

        @property
        def ansifunction(self) -> Type[AnsiFunction[Any]]:
            ...

        @property
        def array_agg(self) -> Type[array_agg[Any]]:
            ...

        @property
        def cast(self) -> Type[Cast[Any]]:
            ...

        @property
        def char_length(self) -> Type[char_length]:
            ...

        @overload
        def coalesce(self, col: ColumnElement[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> coalesce[_T]:
            ...

        @overload
        def coalesce(self, col: _ColumnExpressionArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> coalesce[_T]:
            ...

        @overload
        def coalesce(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> coalesce[_T]:
            ...

        def coalesce(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> coalesce[_T]:
            ...

        @property
        def concat(self) -> Type[concat]:
            ...

        @property
        def count(self) -> Type[count]:
            ...

        @property
        def cube(self) -> Type[cube[Any]]:
            ...

        @property
        def cume_dist(self) -> Type[cume_dist]:
            ...

        @property
        def current_date(self) -> Type[current_date]:
            ...

        @property
        def current_time(self) -> Type[current_time]:
            ...

        @property
        def current_timestamp(self) -> Type[current_timestamp]:
            ...

        @property
        def current_user(self) -> Type[current_user]:
            ...

        @property
        def dense_rank(self) -> Type[dense_rank]:
            ...

        @property
        def extract(self) -> Type[Extract]:
            ...

        @property
        def grouping_sets(self) -> Type[grouping_sets[Any]]:
            ...

        @property
        def localtime(self) -> Type[localtime]:
            ...

        @property
        def localtimestamp(self) -> Type[localtimestamp]:
            ...

        @overload
        def max(self, col: ColumnElement[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> max[_T]:
            ...

        @overload
        def max(self, col: _ColumnExpressionArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> max[_T]:
            ...

        @overload
        def max(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> max[_T]:
            ...

        def max(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> max[_T]:
            ...

        @overload
        def min(self, col: ColumnElement[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> min[_T]:
            ...

        @overload
        def min(self, col: _ColumnExpressionArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> min[_T]:
            ...

        @overload
        def min(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> min[_T]:
            ...

        def min(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> min[_T]:
            ...

        @property
        def mode(self) -> Type[mode[Any]]:
            ...

        @property
        def next_value(self) -> Type[next_value]:
            ...

        @property
        def now(self) -> Type[now]:
            ...

        @property
        def orderedsetagg(self) -> Type[OrderedSetAgg[Any]]:
            ...

        @property
        def percent_rank(self) -> Type[percent_rank]:
            ...

        @property
        def percentile_cont(self) -> Type[percentile_cont[Any]]:
            ...

        @property
        def percentile_disc(self) -> Type[percentile_disc[Any]]:
            ...

        @property
        def random(self) -> Type[random]:
            ...

        @property
        def rank(self) -> Type[rank]:
            ...

        @property
        def rollup(self) -> Type[rollup[Any]]:
            ...

        @property
        def session_user(self) -> Type[session_user]:
            ...

        @overload
        def sum(self, col: ColumnElement[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> sum[_T]:
            ...

        @overload
        def sum(self, col: _ColumnExpressionArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> sum[_T]:
            ...

        @overload
        def sum(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> sum[_T]:
            ...

        def sum(self, col: _ColumnExpressionOrLiteralArgument[_T], *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any) -> sum[_T]:
            ...

        @property
        def sysdate(self) -> Type[sysdate]:
            ...

        @property
        def user(self) -> Type[user]:
            ...