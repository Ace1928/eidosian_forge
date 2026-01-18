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
class TypeEngine(Visitable, Generic[_T]):
    """The ultimate base class for all SQL datatypes.

    Common subclasses of :class:`.TypeEngine` include
    :class:`.String`, :class:`.Integer`, and :class:`.Boolean`.

    For an overview of the SQLAlchemy typing system, see
    :ref:`types_toplevel`.

    .. seealso::

        :ref:`types_toplevel`

    """
    _sqla_type = True
    _isnull = False
    _is_tuple_type = False
    _is_table_value = False
    _is_array = False
    _is_type_decorator = False
    render_bind_cast = False
    'Render bind casts for :attr:`.BindTyping.RENDER_CASTS` mode.\n\n    If True, this type (usually a dialect level impl type) signals\n    to the compiler that a cast should be rendered around a bound parameter\n    for this type.\n\n    .. versionadded:: 2.0\n\n    .. seealso::\n\n        :class:`.BindTyping`\n\n    '
    render_literal_cast = False
    'render casts when rendering a value as an inline literal,\n    e.g. with :meth:`.TypeEngine.literal_processor`.\n\n    .. versionadded:: 2.0\n\n    '

    class Comparator(ColumnOperators, Generic[_CT]):
        """Base class for custom comparison operations defined at the
        type level.  See :attr:`.TypeEngine.comparator_factory`.


        """
        __slots__ = ('expr', 'type')
        expr: ColumnElement[_CT]
        type: TypeEngine[_CT]

        def __clause_element__(self) -> ColumnElement[_CT]:
            return self.expr

        def __init__(self, expr: ColumnElement[_CT]):
            self.expr = expr
            self.type = expr.type

        @util.preload_module('sqlalchemy.sql.default_comparator')
        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            default_comparator = util.preloaded.sql_default_comparator
            op_fn, addtl_kw = default_comparator.operator_lookup[op.__name__]
            if kwargs:
                addtl_kw = addtl_kw.union(kwargs)
            return op_fn(self.expr, op, *other, **addtl_kw)

        @util.preload_module('sqlalchemy.sql.default_comparator')
        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            default_comparator = util.preloaded.sql_default_comparator
            op_fn, addtl_kw = default_comparator.operator_lookup[op.__name__]
            if kwargs:
                addtl_kw = addtl_kw.union(kwargs)
            return op_fn(self.expr, op, other, reverse=True, **addtl_kw)

        def _adapt_expression(self, op: OperatorType, other_comparator: TypeEngine.Comparator[Any]) -> Tuple[OperatorType, TypeEngine[Any]]:
            """evaluate the return type of <self> <op> <othertype>,
            and apply any adaptations to the given operator.

            This method determines the type of a resulting binary expression
            given two source types and an operator.   For example, two
            :class:`_schema.Column` objects, both of the type
            :class:`.Integer`, will
            produce a :class:`.BinaryExpression` that also has the type
            :class:`.Integer` when compared via the addition (``+``) operator.
            However, using the addition operator with an :class:`.Integer`
            and a :class:`.Date` object will produce a :class:`.Date`, assuming
            "days delta" behavior by the database (in reality, most databases
            other than PostgreSQL don't accept this particular operation).

            The method returns a tuple of the form <operator>, <type>.
            The resulting operator and type will be those applied to the
            resulting :class:`.BinaryExpression` as the final operator and the
            right-hand side of the expression.

            Note that only a subset of operators make usage of
            :meth:`._adapt_expression`,
            including math operators and user-defined operators, but not
            boolean comparison or special SQL keywords like MATCH or BETWEEN.

            """
            return (op, self.type)
    hashable = True
    "Flag, if False, means values from this type aren't hashable.\n\n    Used by the ORM when uniquing result lists.\n\n    "
    comparator_factory: _ComparatorFactory[Any] = Comparator
    'A :class:`.TypeEngine.Comparator` class which will apply\n    to operations performed by owning :class:`_expression.ColumnElement`\n    objects.\n\n    The :attr:`.comparator_factory` attribute is a hook consulted by\n    the core expression system when column and SQL expression operations\n    are performed.   When a :class:`.TypeEngine.Comparator` class is\n    associated with this attribute, it allows custom re-definition of\n    all existing operators, as well as definition of new operators.\n    Existing operators include those provided by Python operator overloading\n    such as :meth:`.operators.ColumnOperators.__add__` and\n    :meth:`.operators.ColumnOperators.__eq__`,\n    those provided as standard\n    attributes of :class:`.operators.ColumnOperators` such as\n    :meth:`.operators.ColumnOperators.like`\n    and :meth:`.operators.ColumnOperators.in_`.\n\n    Rudimentary usage of this hook is allowed through simple subclassing\n    of existing types, or alternatively by using :class:`.TypeDecorator`.\n    See the documentation section :ref:`types_operators` for examples.\n\n    '
    sort_key_function: Optional[Callable[[Any], Any]] = None
    'A sorting function that can be passed as the key to sorted.\n\n    The default value of ``None`` indicates that the values stored by\n    this type are self-sorting.\n\n    .. versionadded:: 1.3.8\n\n    '
    should_evaluate_none: bool = False
    "If True, the Python constant ``None`` is considered to be handled\n    explicitly by this type.\n\n    The ORM uses this flag to indicate that a positive value of ``None``\n    is passed to the column in an INSERT statement, rather than omitting\n    the column from the INSERT statement which has the effect of firing\n    off column-level defaults.   It also allows types which have special\n    behavior for Python None, such as a JSON type, to indicate that\n    they'd like to handle the None value explicitly.\n\n    To set this flag on an existing type, use the\n    :meth:`.TypeEngine.evaluates_none` method.\n\n    .. seealso::\n\n        :meth:`.TypeEngine.evaluates_none`\n\n    "
    _variant_mapping: util.immutabledict[str, TypeEngine[Any]] = util.EMPTY_DICT

    def evaluates_none(self) -> Self:
        """Return a copy of this type which has the
        :attr:`.should_evaluate_none` flag set to True.

        E.g.::

                Table(
                    'some_table', metadata,
                    Column(
                        String(50).evaluates_none(),
                        nullable=True,
                        server_default='no value')
                )

        The ORM uses this flag to indicate that a positive value of ``None``
        is passed to the column in an INSERT statement, rather than omitting
        the column from the INSERT statement which has the effect of firing
        off column-level defaults.   It also allows for types which have
        special behavior associated with the Python None value to indicate
        that the value doesn't necessarily translate into SQL NULL; a
        prime example of this is a JSON type which may wish to persist the
        JSON value ``'null'``.

        In all cases, the actual NULL SQL value can be always be
        persisted in any column by using
        the :obj:`_expression.null` SQL construct in an INSERT statement
        or associated with an ORM-mapped attribute.

        .. note::

            The "evaluates none" flag does **not** apply to a value
            of ``None`` passed to :paramref:`_schema.Column.default` or
            :paramref:`_schema.Column.server_default`; in these cases,
            ``None``
            still means "no default".

        .. seealso::

            :ref:`session_forcing_null` - in the ORM documentation

            :paramref:`.postgresql.JSON.none_as_null` - PostgreSQL JSON
            interaction with this flag.

            :attr:`.TypeEngine.should_evaluate_none` - class-level flag

        """
        typ = self.copy()
        typ.should_evaluate_none = True
        return typ

    def copy(self, **kw: Any) -> Self:
        return self.adapt(self.__class__)

    def copy_value(self, value: Any) -> Any:
        return value

    def literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[_T]]:
        """Return a conversion function for processing literal values that are
        to be rendered directly without using binds.

        This function is used when the compiler makes use of the
        "literal_binds" flag, typically used in DDL generation as well
        as in certain scenarios where backends don't accept bound parameters.

        Returns a callable which will receive a literal Python value
        as the sole positional argument and will return a string representation
        to be rendered in a SQL statement.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.literal_processor`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.literal_processor`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.process_literal_param`.

            .. seealso::

                :ref:`types_typedecorator`


        """
        return None

    def bind_processor(self, dialect: Dialect) -> Optional[_BindProcessorType[_T]]:
        """Return a conversion function for processing bind values.

        Returns a callable which will receive a bind parameter value
        as the sole positional argument and will return a value to
        send to the DB-API.

        If processing is not necessary, the method should return ``None``.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.bind_processor`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.bind_processor`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.process_bind_param`.

            .. seealso::

                :ref:`types_typedecorator`


        :param dialect: Dialect instance in use.

        """
        return None

    def result_processor(self, dialect: Dialect, coltype: object) -> Optional[_ResultProcessorType[_T]]:
        """Return a conversion function for processing result row values.

        Returns a callable which will receive a result row column
        value as the sole positional argument and will return a value
        to return to the user.

        If processing is not necessary, the method should return ``None``.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.result_processor`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.result_processor`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.process_result_value`.

            .. seealso::

                :ref:`types_typedecorator`

        :param dialect: Dialect instance in use.

        :param coltype: DBAPI coltype argument received in cursor.description.

        """
        return None

    def column_expression(self, colexpr: ColumnElement[_T]) -> Optional[ColumnElement[_T]]:
        """Given a SELECT column expression, return a wrapping SQL expression.

        This is typically a SQL function that wraps a column expression
        as rendered in the columns clause of a SELECT statement.
        It is used for special data types that require
        columns to be wrapped in some special database function in order
        to coerce the value before being sent back to the application.
        It is the SQL analogue of the :meth:`.TypeEngine.result_processor`
        method.

        This method is called during the **SQL compilation** phase of a
        statement, when rendering a SQL string. It is **not** called
        against specific values.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.column_expression`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.column_expression`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.column_expression`.

            .. seealso::

                :ref:`types_typedecorator`


        .. seealso::

            :ref:`types_sql_value_processing`

        """
        return None

    @util.memoized_property
    def _has_column_expression(self) -> bool:
        """memoized boolean, check if column_expression is implemented.

        Allows the method to be skipped for the vast majority of expression
        types that don't use this feature.

        """
        return self.__class__.column_expression.__code__ is not TypeEngine.column_expression.__code__

    def bind_expression(self, bindvalue: BindParameter[_T]) -> Optional[ColumnElement[_T]]:
        """Given a bind value (i.e. a :class:`.BindParameter` instance),
        return a SQL expression in its place.

        This is typically a SQL function that wraps the existing bound
        parameter within the statement.  It is used for special data types
        that require literals being wrapped in some special database function
        in order to coerce an application-level value into a database-specific
        format.  It is the SQL analogue of the
        :meth:`.TypeEngine.bind_processor` method.

        This method is called during the **SQL compilation** phase of a
        statement, when rendering a SQL string. It is **not** called
        against specific values.

        Note that this method, when implemented, should always return
        the exact same structure, without any conditional logic, as it
        may be used in an executemany() call against an arbitrary number
        of bound parameter sets.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.bind_expression`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.bind_expression`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.bind_expression`.

            .. seealso::

                :ref:`types_typedecorator`

        .. seealso::

            :ref:`types_sql_value_processing`

        """
        return None

    @util.memoized_property
    def _has_bind_expression(self) -> bool:
        """memoized boolean, check if bind_expression is implemented.

        Allows the method to be skipped for the vast majority of expression
        types that don't use this feature.

        """
        return util.method_is_overridden(self, TypeEngine.bind_expression)

    @staticmethod
    def _to_instance(cls_or_self: Union[Type[_TE], _TE]) -> _TE:
        return to_instance(cls_or_self)

    def compare_values(self, x: Any, y: Any) -> bool:
        """Compare two values for equality."""
        return x == y

    def get_dbapi_type(self, dbapi: ModuleType) -> Optional[Any]:
        """Return the corresponding type object from the underlying DB-API, if
        any.

        This can be useful for calling ``setinputsizes()``, for example.

        """
        return None

    @property
    def python_type(self) -> Type[Any]:
        """Return the Python type object expected to be returned
        by instances of this type, if known.

        Basically, for those types which enforce a return type,
        or are known across the board to do such for all common
        DBAPIs (like ``int`` for example), will return that type.

        If a return type is not defined, raises
        ``NotImplementedError``.

        Note that any type also accommodates NULL in SQL which
        means you can also get back ``None`` from any type
        in practice.

        """
        raise NotImplementedError()

    def with_variant(self, type_: _TypeEngineArgument[Any], *dialect_names: str) -> Self:
        """Produce a copy of this type object that will utilize the given
        type when applied to the dialect of the given name.

        e.g.::

            from sqlalchemy.types import String
            from sqlalchemy.dialects import mysql

            string_type = String()

            string_type = string_type.with_variant(
                mysql.VARCHAR(collation='foo'), 'mysql', 'mariadb'
            )

        The variant mapping indicates that when this type is
        interpreted by a specific dialect, it will instead be
        transmuted into the given type, rather than using the
        primary type.

        .. versionchanged:: 2.0 the :meth:`_types.TypeEngine.with_variant`
           method now works with a :class:`_types.TypeEngine` object "in
           place", returning a copy of the original type rather than returning
           a wrapping object; the ``Variant`` class is no longer used.

        :param type\\_: a :class:`.TypeEngine` that will be selected
         as a variant from the originating type, when a dialect
         of the given name is in use.
        :param \\*dialect_names: one or more base names of the dialect which
         uses this type. (i.e. ``'postgresql'``, ``'mysql'``, etc.)

         .. versionchanged:: 2.0 multiple dialect names can be specified
            for one variant.

        .. seealso::

            :ref:`types_with_variant` - illustrates the use of
            :meth:`_types.TypeEngine.with_variant`.

        """
        if not dialect_names:
            raise exc.ArgumentError('At least one dialect name is required')
        for dialect_name in dialect_names:
            if dialect_name in self._variant_mapping:
                raise exc.ArgumentError(f'Dialect {dialect_name!r} is already present in the mapping for this {self!r}')
        new_type = self.copy()
        type_ = to_instance(type_)
        if type_._variant_mapping:
            raise exc.ArgumentError("can't pass a type that already has variants as a dialect-level type to with_variant()")
        new_type._variant_mapping = self._variant_mapping.union({dialect_name: type_ for dialect_name in dialect_names})
        return new_type

    def _resolve_for_literal(self, value: Any) -> Self:
        """adjust this type given a literal Python value that will be
        stored in a bound parameter.

        Used exclusively by _resolve_value_to_type().

        .. versionadded:: 1.4.30 or 2.0

        TODO: this should be part of public API

        .. seealso::

            :meth:`.TypeEngine._resolve_for_python_type`

        """
        return self

    def _resolve_for_python_type(self, python_type: Type[Any], matched_on: _MatchedOnType, matched_on_flattened: Type[Any]) -> Optional[Self]:
        """given a Python type (e.g. ``int``, ``str``, etc. ) return an
        instance of this :class:`.TypeEngine` that's appropriate for this type.

        An additional argument ``matched_on`` is passed, which indicates an
        entry from the ``__mro__`` of the given ``python_type`` that more
        specifically matches how the caller located this :class:`.TypeEngine`
        object.   Such as, if a lookup of some kind links the ``int`` Python
        type to the :class:`.Integer` SQL type, and the original object
        was some custom subclass of ``int`` such as ``MyInt(int)``, the
        arguments passed would be ``(MyInt, int)``.

        If the given Python type does not correspond to this
        :class:`.TypeEngine`, or the Python type is otherwise ambiguous, the
        method should return None.

        For simple cases, the method checks that the ``python_type``
        and ``matched_on`` types are the same (i.e. not a subclass), and
        returns self; for all other cases, it returns ``None``.

        The initial use case here is for the ORM to link user-defined
        Python standard library ``enum.Enum`` classes to the SQLAlchemy
        :class:`.Enum` SQL type when constructing ORM Declarative mappings.

        :param python_type: the Python type we want to use
        :param matched_on: the Python type that led us to choose this
         particular :class:`.TypeEngine` class, which would be a supertype
         of ``python_type``.   By default, the request is rejected if
         ``python_type`` doesn't match ``matched_on`` (None is returned).

        .. versionadded:: 2.0.0b4

        TODO: this should be part of public API

        .. seealso::

            :meth:`.TypeEngine._resolve_for_literal`

        """
        if python_type is not matched_on_flattened:
            return None
        return self

    @util.ro_memoized_property
    def _type_affinity(self) -> Optional[Type[TypeEngine[_T]]]:
        """Return a rudimental 'affinity' value expressing the general class
        of type."""
        typ = None
        for t in self.__class__.__mro__:
            if t is TypeEngine or TypeEngineMixin in t.__bases__:
                return typ
            elif issubclass(t, TypeEngine):
                typ = t
        else:
            return self.__class__

    @util.ro_memoized_property
    def _generic_type_affinity(self) -> Type[TypeEngine[_T]]:
        best_camelcase = None
        best_uppercase = None
        if not isinstance(self, TypeEngine):
            return self.__class__
        for t in self.__class__.__mro__:
            if t.__module__ in ('sqlalchemy.sql.sqltypes', 'sqlalchemy.sql.type_api') and issubclass(t, TypeEngine) and (TypeEngineMixin not in t.__bases__) and (t not in (TypeEngine, TypeEngineMixin)) and (t.__name__[0] != '_'):
                if t.__name__.isupper() and (not best_uppercase):
                    best_uppercase = t
                elif not t.__name__.isupper() and (not best_camelcase):
                    best_camelcase = t
        return best_camelcase or best_uppercase or cast('Type[TypeEngine[_T]]', NULLTYPE.__class__)

    def as_generic(self, allow_nulltype: bool=False) -> TypeEngine[_T]:
        """
        Return an instance of the generic type corresponding to this type
        using heuristic rule. The method may be overridden if this
        heuristic rule is not sufficient.

        >>> from sqlalchemy.dialects.mysql import INTEGER
        >>> INTEGER(display_width=4).as_generic()
        Integer()

        >>> from sqlalchemy.dialects.mysql import NVARCHAR
        >>> NVARCHAR(length=100).as_generic()
        Unicode(length=100)

        .. versionadded:: 1.4.0b2


        .. seealso::

            :ref:`metadata_reflection_dbagnostic_types` - describes the
            use of :meth:`_types.TypeEngine.as_generic` in conjunction with
            the :meth:`_sql.DDLEvents.column_reflect` event, which is its
            intended use.

        """
        if not allow_nulltype and self._generic_type_affinity == NULLTYPE.__class__:
            raise NotImplementedError('Default TypeEngine.as_generic() heuristic method was unsuccessful for {}. A custom as_generic() method must be implemented for this type class.'.format(self.__class__.__module__ + '.' + self.__class__.__name__))
        return util.constructor_copy(self, self._generic_type_affinity)

    def dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        """Return a dialect-specific implementation for this
        :class:`.TypeEngine`.

        """
        try:
            tm = dialect._type_memos[self]
        except KeyError:
            pass
        else:
            return tm['impl']
        return self._dialect_info(dialect)['impl']

    def _unwrapped_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        """Return the 'unwrapped' dialect impl for this type.

        For a type that applies wrapping logic (e.g. TypeDecorator), give
        us the real, actual dialect-level type that is used.

        This is used by TypeDecorator itself as well at least one case where
        dialects need to check that a particular specific dialect-level
        type is in use, within the :meth:`.DefaultDialect.set_input_sizes`
        method.

        """
        return self.dialect_impl(dialect)

    def _cached_literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[_T]]:
        """Return a dialect-specific literal processor for this type."""
        try:
            return dialect._type_memos[self]['literal']
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        d['literal'] = lp = d['impl'].literal_processor(dialect)
        return lp

    def _cached_bind_processor(self, dialect: Dialect) -> Optional[_BindProcessorType[_T]]:
        """Return a dialect-specific bind processor for this type."""
        try:
            return dialect._type_memos[self]['bind']
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        d['bind'] = bp = d['impl'].bind_processor(dialect)
        return bp

    def _cached_result_processor(self, dialect: Dialect, coltype: Any) -> Optional[_ResultProcessorType[_T]]:
        """Return a dialect-specific result processor for this type."""
        try:
            return dialect._type_memos[self]['result'][coltype]
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        rp = d['impl'].result_processor(dialect, coltype)
        d['result'][coltype] = rp
        return rp

    def _cached_custom_processor(self, dialect: Dialect, key: str, fn: Callable[[TypeEngine[_T]], _O]) -> _O:
        """return a dialect-specific processing object for
        custom purposes.

        The cx_Oracle dialect uses this at the moment.

        """
        try:
            return cast(_O, dialect._type_memos[self]['custom'][key])
        except KeyError:
            pass
        d = self._dialect_info(dialect)
        impl = d['impl']
        custom_dict = d.setdefault('custom', {})
        custom_dict[key] = result = fn(impl)
        return result

    def _dialect_info(self, dialect: Dialect) -> _TypeMemoDict:
        """Return a dialect-specific registry which
        caches a dialect-specific implementation, bind processing
        function, and one or more result processing functions."""
        if self in dialect._type_memos:
            return dialect._type_memos[self]
        else:
            impl = self._gen_dialect_impl(dialect)
            if impl is self:
                impl = self.adapt(type(self))
            assert impl is not self
            d: _TypeMemoDict = {'impl': impl, 'result': {}}
            dialect._type_memos[self] = d
            return d

    def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name in self._variant_mapping:
            return self._variant_mapping[dialect.name]._gen_dialect_impl(dialect)
        else:
            return dialect.type_descriptor(self)

    @util.memoized_property
    def _static_cache_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
        names = util.get_cls_kwargs(self.__class__)
        return (self.__class__,) + tuple(((k, self.__dict__[k]._static_cache_key if isinstance(self.__dict__[k], TypeEngine) else self.__dict__[k]) for k in names if k in self.__dict__ and (not k.startswith('_')) and (self.__dict__[k] is not None)))

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
        ...

    @overload
    def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        ...

    def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        """Produce an "adapted" form of this type, given an "impl" class
        to work with.

        This method is used internally to associate generic
        types with "implementation" types that are specific to a particular
        dialect.
        """
        typ = util.constructor_copy(self, cast(Type[TypeEngine[Any]], cls), **kw)
        typ._variant_mapping = self._variant_mapping
        return typ

    def coerce_compared_value(self, op: Optional[OperatorType], value: Any) -> TypeEngine[Any]:
        """Suggest a type for a 'coerced' Python value in an expression.

        Given an operator and value, gives the type a chance
        to return a type which the value should be coerced into.

        The default behavior here is conservative; if the right-hand
        side is already coerced into a SQL type based on its
        Python type, it is usually left alone.

        End-user functionality extension here should generally be via
        :class:`.TypeDecorator`, which provides more liberal behavior in that
        it defaults to coercing the other side of the expression into this
        type, thus applying special Python conversions above and beyond those
        needed by the DBAPI to both ides. It also provides the public method
        :meth:`.TypeDecorator.coerce_compared_value` which is intended for
        end-user customization of this behavior.

        """
        _coerced_type = _resolve_value_to_type(value)
        if _coerced_type is NULLTYPE or _coerced_type._type_affinity is self._type_affinity:
            return self
        else:
            return _coerced_type

    def _compare_type_affinity(self, other: TypeEngine[Any]) -> bool:
        return self._type_affinity is other._type_affinity

    def compile(self, dialect: Optional[Dialect]=None) -> str:
        """Produce a string-compiled form of this :class:`.TypeEngine`.

        When called with no arguments, uses a "default" dialect
        to produce a string result.

        :param dialect: a :class:`.Dialect` instance.

        """
        if dialect is None:
            dialect = self._default_dialect()
        return dialect.type_compiler_instance.process(self)

    @util.preload_module('sqlalchemy.engine.default')
    def _default_dialect(self) -> Dialect:
        default = util.preloaded.engine_default
        return default.StrCompileDialect()

    def __str__(self) -> str:
        return str(self.compile())

    def __repr__(self) -> str:
        return util.generic_repr(self)