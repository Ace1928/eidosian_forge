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
class ColumnElement(roles.ColumnArgumentOrKeyRole, roles.StatementOptionRole, roles.WhereHavingRole, roles.BinaryElementRole[_T], roles.OrderByRole, roles.ColumnsClauseRole, roles.LimitOffsetRole, roles.DMLColumnRole, roles.DDLConstraintColumnRole, roles.DDLExpressionRole, SQLColumnExpression[_T], DQLDMLClauseElement):
    """Represent a column-oriented SQL expression suitable for usage in the
    "columns" clause, WHERE clause etc. of a statement.

    While the most familiar kind of :class:`_expression.ColumnElement` is the
    :class:`_schema.Column` object, :class:`_expression.ColumnElement`
    serves as the basis
    for any unit that may be present in a SQL expression, including
    the expressions themselves, SQL functions, bound parameters,
    literal expressions, keywords such as ``NULL``, etc.
    :class:`_expression.ColumnElement`
    is the ultimate base class for all such elements.

    A wide variety of SQLAlchemy Core functions work at the SQL expression
    level, and are intended to accept instances of
    :class:`_expression.ColumnElement` as
    arguments.  These functions will typically document that they accept a
    "SQL expression" as an argument.  What this means in terms of SQLAlchemy
    usually refers to an input which is either already in the form of a
    :class:`_expression.ColumnElement` object,
    or a value which can be **coerced** into
    one.  The coercion rules followed by most, but not all, SQLAlchemy Core
    functions with regards to SQL expressions are as follows:

        * a literal Python value, such as a string, integer or floating
          point value, boolean, datetime, ``Decimal`` object, or virtually
          any other Python object, will be coerced into a "literal bound
          value".  This generally means that a :func:`.bindparam` will be
          produced featuring the given value embedded into the construct; the
          resulting :class:`.BindParameter` object is an instance of
          :class:`_expression.ColumnElement`.
          The Python value will ultimately be sent
          to the DBAPI at execution time as a parameterized argument to the
          ``execute()`` or ``executemany()`` methods, after SQLAlchemy
          type-specific converters (e.g. those provided by any associated
          :class:`.TypeEngine` objects) are applied to the value.

        * any special object value, typically ORM-level constructs, which
          feature an accessor called ``__clause_element__()``.  The Core
          expression system looks for this method when an object of otherwise
          unknown type is passed to a function that is looking to coerce the
          argument into a :class:`_expression.ColumnElement` and sometimes a
          :class:`_expression.SelectBase` expression.
          It is used within the ORM to
          convert from ORM-specific objects like mapped classes and
          mapped attributes into Core expression objects.

        * The Python ``None`` value is typically interpreted as ``NULL``,
          which in SQLAlchemy Core produces an instance of :func:`.null`.

    A :class:`_expression.ColumnElement` provides the ability to generate new
    :class:`_expression.ColumnElement`
    objects using Python expressions.  This means that Python operators
    such as ``==``, ``!=`` and ``<`` are overloaded to mimic SQL operations,
    and allow the instantiation of further :class:`_expression.ColumnElement`
    instances
    which are composed from other, more fundamental
    :class:`_expression.ColumnElement`
    objects.  For example, two :class:`.ColumnClause` objects can be added
    together with the addition operator ``+`` to produce
    a :class:`.BinaryExpression`.
    Both :class:`.ColumnClause` and :class:`.BinaryExpression` are subclasses
    of :class:`_expression.ColumnElement`:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy.sql import column
        >>> column('a') + column('b')
        <sqlalchemy.sql.expression.BinaryExpression object at 0x101029dd0>
        >>> print(column('a') + column('b'))
        {printsql}a + b

    .. seealso::

        :class:`_schema.Column`

        :func:`_expression.column`

    """
    __visit_name__ = 'column_element'
    primary_key: bool = False
    _is_clone_of: Optional[ColumnElement[_T]]
    _is_column_element = True
    _insert_sentinel: bool = False
    _omit_from_statements = False
    _is_collection_aggregate = False
    foreign_keys: AbstractSet[ForeignKey] = frozenset()

    @util.memoized_property
    def _proxies(self) -> List[ColumnElement[Any]]:
        return []

    @util.non_memoized_property
    def _tq_label(self) -> Optional[str]:
        """The named label that can be used to target
        this column in a result set in a "table qualified" context.

        This label is almost always the label used when
        rendering <expr> AS <label> in a SELECT statement when using
        the LABEL_STYLE_TABLENAME_PLUS_COL label style, which is what the
        legacy ORM ``Query`` object uses as well.

        For a regular Column bound to a Table, this is typically the label
        <tablename>_<columnname>.  For other constructs, different rules
        may apply, such as anonymized labels and others.

        .. versionchanged:: 1.4.21 renamed from ``._label``

        """
        return None
    key: Optional[str] = None
    'The \'key\' that in some circumstances refers to this object in a\n    Python namespace.\n\n    This typically refers to the "key" of the column as present in the\n    ``.c`` collection of a selectable, e.g. ``sometable.c["somekey"]`` would\n    return a :class:`_schema.Column` with a ``.key`` of "somekey".\n\n    '

    @HasMemoized.memoized_attribute
    def _tq_key_label(self) -> Optional[str]:
        """A label-based version of 'key' that in some circumstances refers
        to this object in a Python namespace.


        _tq_key_label comes into play when a select() statement is constructed
        with apply_labels(); in this case, all Column objects in the ``.c``
        collection are rendered as <tablename>_<columnname> in SQL; this is
        essentially the value of ._label. But to locate those columns in the
        ``.c`` collection, the name is along the lines of <tablename>_<key>;
        that's the typical value of .key_label.

        .. versionchanged:: 1.4.21 renamed from ``._key_label``

        """
        return self._proxy_key

    @property
    def _key_label(self) -> Optional[str]:
        """legacy; renamed to _tq_key_label"""
        return self._tq_key_label

    @property
    def _label(self) -> Optional[str]:
        """legacy; renamed to _tq_label"""
        return self._tq_label

    @property
    def _non_anon_label(self) -> Optional[str]:
        """the 'name' that naturally applies this element when rendered in
        SQL.

        Concretely, this is the "name" of a column or a label in a
        SELECT statement; ``<columnname>`` and ``<labelname>`` below::

            SELECT <columnmame> FROM table

            SELECT column AS <labelname> FROM table

        Above, the two names noted will be what's present in the DBAPI
        ``cursor.description`` as the names.

        If this attribute returns ``None``, it means that the SQL element as
        written does not have a 100% fully predictable "name" that would appear
        in the ``cursor.description``. Examples include SQL functions, CAST
        functions, etc. While such things do return names in
        ``cursor.description``, they are only predictable on a
        database-specific basis; e.g. an expression like ``MAX(table.col)`` may
        appear as the string ``max`` on one database (like PostgreSQL) or may
        appear as the whole expression ``max(table.col)`` on SQLite.

        The default implementation looks for a ``.name`` attribute on the
        object, as has been the precedent established in SQLAlchemy for many
        years.  An exception is made on the ``FunctionElement`` subclass
        so that the return value is always ``None``.

        .. versionadded:: 1.4.21



        """
        return getattr(self, 'name', None)
    _render_label_in_columns_clause = True
    'A flag used by select._columns_plus_names that helps to determine\n    we are actually going to render in terms of "SELECT <col> AS <label>".\n    This flag can be returned as False for some Column objects that want\n    to be rendered as simple "SELECT <col>"; typically columns that don\'t have\n    any parent table and are named the same as what the label would be\n    in any case.\n\n    '
    _allow_label_resolve = True
    'A flag that can be flipped to prevent a column from being resolvable\n    by string label name.\n\n    The joined eager loader strategy in the ORM uses this, for example.\n\n    '
    _is_implicitly_boolean = False
    _alt_names: Sequence[str] = ()

    @overload
    def self_group(self: ColumnElement[_T], against: Optional[OperatorType]=None) -> ColumnElement[_T]:
        ...

    @overload
    def self_group(self: ColumnElement[Any], against: Optional[OperatorType]=None) -> ColumnElement[Any]:
        ...

    def self_group(self, against: Optional[OperatorType]=None) -> ColumnElement[Any]:
        if against in (operators.and_, operators.or_, operators._asbool) and self.type._type_affinity is type_api.BOOLEANTYPE._type_affinity:
            return AsBoolean(self, operators.is_true, operators.is_false)
        elif against in (operators.any_op, operators.all_op):
            return Grouping(self)
        else:
            return self

    @overload
    def _negate(self: ColumnElement[bool]) -> ColumnElement[bool]:
        ...

    @overload
    def _negate(self: ColumnElement[_T]) -> ColumnElement[_T]:
        ...

    def _negate(self) -> ColumnElement[Any]:
        if self.type._type_affinity is type_api.BOOLEANTYPE._type_affinity:
            return AsBoolean(self, operators.is_false, operators.is_true)
        else:
            grouped = self.self_group(against=operators.inv)
            assert isinstance(grouped, ColumnElement)
            return UnaryExpression(grouped, operator=operators.inv, wraps_column_expression=True)
    type: TypeEngine[_T]
    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:
            return type_api.NULLTYPE

    @HasMemoized.memoized_attribute
    def comparator(self) -> TypeEngine.Comparator[_T]:
        try:
            comparator_factory = self.type.comparator_factory
        except AttributeError as err:
            raise TypeError("Object %r associated with '.type' attribute is not a TypeEngine class or object" % self.type) from err
        else:
            return comparator_factory(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, key: str) -> Any:
        try:
            return getattr(self.comparator, key)
        except AttributeError as err:
            raise AttributeError('Neither %r object nor %r object has an attribute %r' % (type(self).__name__, type(self.comparator).__name__, key)) from err

    def operate(self, op: operators.OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(self.comparator, *other, **kwargs)

    def reverse_operate(self, op: operators.OperatorType, other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(other, self.comparator, **kwargs)

    def _bind_param(self, operator: operators.OperatorType, obj: Any, type_: Optional[TypeEngine[_T]]=None, expanding: bool=False) -> BindParameter[_T]:
        return BindParameter(None, obj, _compared_to_operator=operator, type_=type_, _compared_to_type=self.type, unique=True, expanding=expanding)

    @property
    def expression(self) -> ColumnElement[Any]:
        """Return a column expression.

        Part of the inspection interface; returns self.

        """
        return self

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    @util.memoized_property
    def base_columns(self) -> FrozenSet[ColumnElement[Any]]:
        return frozenset((c for c in self.proxy_set if not c._proxies))

    @util.memoized_property
    def proxy_set(self) -> FrozenSet[ColumnElement[Any]]:
        """set of all columns we are proxying

        as of 2.0 this is explicitly deannotated columns.  previously it was
        effectively deannotated columns but wasn't enforced.  annotated
        columns should basically not go into sets if at all possible because
        their hashing behavior is very non-performant.

        """
        return frozenset([self._deannotate()]).union(itertools.chain(*[c.proxy_set for c in self._proxies]))

    @util.memoized_property
    def _expanded_proxy_set(self) -> FrozenSet[ColumnElement[Any]]:
        return frozenset(_expand_cloned(self.proxy_set))

    def _uncached_proxy_list(self) -> List[ColumnElement[Any]]:
        """An 'uncached' version of proxy set.

        This list includes annotated columns which perform very poorly in
        set operations.

        """
        return [self] + list(itertools.chain(*[c._uncached_proxy_list() for c in self._proxies]))

    def shares_lineage(self, othercolumn: ColumnElement[Any]) -> bool:
        """Return True if the given :class:`_expression.ColumnElement`
        has a common ancestor to this :class:`_expression.ColumnElement`."""
        return bool(self.proxy_set.intersection(othercolumn.proxy_set))

    def _compare_name_for_result(self, other: ColumnElement[Any]) -> bool:
        """Return True if the given column element compares to this one
        when targeting within a result row."""
        return hasattr(other, 'name') and hasattr(self, 'name') and (other.name == self.name)

    @HasMemoized.memoized_attribute
    def _proxy_key(self) -> Optional[str]:
        if self._annotations and 'proxy_key' in self._annotations:
            return cast(str, self._annotations['proxy_key'])
        name = self.key
        if not name:
            name = self._non_anon_label
        if isinstance(name, _anonymous_label):
            return None
        else:
            return name

    @HasMemoized.memoized_attribute
    def _expression_label(self) -> Optional[str]:
        """a suggested label to use in the case that the column has no name,
        which should be used if possible as the explicit 'AS <label>'
        where this expression would normally have an anon label.

        this is essentially mostly what _proxy_key does except it returns
        None if the column has a normal name that can be used.

        """
        if getattr(self, 'name', None) is not None:
            return None
        elif self._annotations and 'proxy_key' in self._annotations:
            return cast(str, self._annotations['proxy_key'])
        else:
            return None

    def _make_proxy(self, selectable: FromClause, *, name: Optional[str]=None, key: Optional[str]=None, name_is_truncatable: bool=False, compound_select_cols: Optional[Sequence[ColumnElement[Any]]]=None, **kw: Any) -> typing_Tuple[str, ColumnClause[_T]]:
        """Create a new :class:`_expression.ColumnElement` representing this
        :class:`_expression.ColumnElement` as it appears in the select list of
        a descending selectable.

        """
        if name is None:
            name = self._anon_name_label
            if key is None:
                key = self._proxy_key
        else:
            key = name
        assert key is not None
        co: ColumnClause[_T] = ColumnClause(coercions.expect(roles.TruncatedLabelRole, name) if name_is_truncatable else name, type_=getattr(self, 'type', None), _selectable=selectable)
        co._propagate_attrs = selectable._propagate_attrs
        if compound_select_cols:
            co._proxies = list(compound_select_cols)
        else:
            co._proxies = [self]
        if selectable._is_clone_of is not None:
            co._is_clone_of = selectable._is_clone_of.columns.get(key)
        return (key, co)

    def cast(self, type_: _TypeEngineArgument[_OPT]) -> Cast[_OPT]:
        """Produce a type cast, i.e. ``CAST(<expression> AS <type>)``.

        This is a shortcut to the :func:`_expression.cast` function.

        .. seealso::

            :ref:`tutorial_casts`

            :func:`_expression.cast`

            :func:`_expression.type_coerce`

        """
        return Cast(self, type_)

    def label(self, name: Optional[str]) -> Label[_T]:
        """Produce a column label, i.e. ``<columnname> AS <name>``.

        This is a shortcut to the :func:`_expression.label` function.

        If 'name' is ``None``, an anonymous label name will be generated.

        """
        return Label(name, self, self.type)

    def _anon_label(self, seed: Optional[str], add_hash: Optional[int]=None) -> _anonymous_label:
        while self._is_clone_of is not None:
            self = self._is_clone_of
        hash_value = hash(self)
        if add_hash:
            assert add_hash < 2 << 15
            assert seed
            hash_value = hash_value << 16 | add_hash
            seed = seed + '_'
        if isinstance(seed, _anonymous_label):
            return _anonymous_label.safe_construct(hash_value, '', enclosing_label=seed)
        return _anonymous_label.safe_construct(hash_value, seed or 'anon')

    @util.memoized_property
    def _anon_name_label(self) -> str:
        """Provides a constant 'anonymous label' for this ColumnElement.

        This is a label() expression which will be named at compile time.
        The same label() is returned each time ``anon_label`` is called so
        that expressions can reference ``anon_label`` multiple times,
        producing the same label name at compile time.

        The compiler uses this function automatically at compile time
        for expressions that are known to be 'unnamed' like binary
        expressions and function calls.

        .. versionchanged:: 1.4.9 - this attribute was not intended to be
           public and is renamed to _anon_name_label.  anon_name exists
           for backwards compat

        """
        name = getattr(self, 'name', None)
        return self._anon_label(name)

    @util.memoized_property
    def _anon_key_label(self) -> _anonymous_label:
        """Provides a constant 'anonymous key label' for this ColumnElement.

        Compare to ``anon_label``, except that the "key" of the column,
        if available, is used to generate the label.

        This is used when a deduplicating key is placed into the columns
        collection of a selectable.

        .. versionchanged:: 1.4.9 - this attribute was not intended to be
           public and is renamed to _anon_key_label.  anon_key_label exists
           for backwards compat

        """
        return self._anon_label(self._proxy_key)

    @property
    @util.deprecated('1.4', 'The :attr:`_expression.ColumnElement.anon_label` attribute is now private, and the public accessor is deprecated.')
    def anon_label(self) -> str:
        return self._anon_name_label

    @property
    @util.deprecated('1.4', 'The :attr:`_expression.ColumnElement.anon_key_label` attribute is now private, and the public accessor is deprecated.')
    def anon_key_label(self) -> str:
        return self._anon_key_label

    def _dedupe_anon_label_idx(self, idx: int) -> str:
        """label to apply to a column that is anon labeled, but repeated
        in the SELECT, so that we have to make an "extra anon" label that
        disambiguates it from the previous appearance.

        these labels come out like "foo_bar_id__1" and have double underscores
        in them.

        """
        label = getattr(self, 'name', None)
        if label is None:
            return self._dedupe_anon_tq_label_idx(idx)
        else:
            return self._anon_label(label, add_hash=idx)

    @util.memoized_property
    def _anon_tq_label(self) -> _anonymous_label:
        return self._anon_label(getattr(self, '_tq_label', None))

    @util.memoized_property
    def _anon_tq_key_label(self) -> _anonymous_label:
        return self._anon_label(getattr(self, '_tq_key_label', None))

    def _dedupe_anon_tq_label_idx(self, idx: int) -> _anonymous_label:
        label = getattr(self, '_tq_label', None) or 'anon'
        return self._anon_label(label, add_hash=idx)