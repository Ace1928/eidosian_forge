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
@inspection._self_inspects
class ClauseElement(SupportsWrappingAnnotations, MemoizedHasCacheKey, HasCopyInternals, ExternallyTraversible, CompilerElement):
    """Base class for elements of a programmatically constructed SQL
    expression.

    """
    __visit_name__ = 'clause'
    if TYPE_CHECKING:

        @util.memoized_property
        def _propagate_attrs(self) -> _PropagateAttrsType:
            """like annotations, however these propagate outwards liberally
            as SQL constructs are built, and are set up at construction time.

            """
            ...
    else:
        _propagate_attrs = util.EMPTY_DICT

    @util.ro_memoized_property
    def description(self) -> Optional[str]:
        return None
    _is_clone_of: Optional[Self] = None
    is_clause_element = True
    is_selectable = False
    is_dml = False
    _is_column_element = False
    _is_keyed_column_element = False
    _is_table = False
    _gen_static_annotations_cache_key = False
    _is_textual = False
    _is_from_clause = False
    _is_returns_rows = False
    _is_text_clause = False
    _is_from_container = False
    _is_select_container = False
    _is_select_base = False
    _is_select_statement = False
    _is_bind_parameter = False
    _is_clause_list = False
    _is_lambda_element = False
    _is_singleton_constant = False
    _is_immutable = False
    _is_star = False

    @property
    def _order_by_label_element(self) -> Optional[Label[Any]]:
        return None
    _cache_key_traversal: _CacheKeyTraversalType = None
    negation_clause: ColumnElement[bool]
    if typing.TYPE_CHECKING:

        def get_children(self, *, omit_attrs: typing_Tuple[str, ...]=..., **kw: Any) -> Iterable[ClauseElement]:
            ...

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return []

    def _set_propagate_attrs(self, values: Mapping[str, Any]) -> Self:
        self._propagate_attrs = util.immutabledict(values)
        return self

    def _clone(self, **kw: Any) -> Self:
        """Create a shallow copy of this ClauseElement.

        This method may be used by a generative API.  Its also used as
        part of the "deep" copy afforded by a traversal that combines
        the _copy_internals() method.

        """
        skip = self._memoized_keys
        c = self.__class__.__new__(self.__class__)
        if skip:
            c.__dict__ = {k: v for k, v in self.__dict__.copy().items() if k not in skip}
        else:
            c.__dict__ = self.__dict__.copy()
        cc = self._is_clone_of
        c._is_clone_of = cc if cc is not None else self
        return c

    def _negate_in_binary(self, negated_op, original_op):
        """a hook to allow the right side of a binary expression to respond
        to a negation of the binary expression.

        Used for the special case of expanding bind parameter with IN.

        """
        return self

    def _with_binary_element_type(self, type_):
        """in the context of binary expression, convert the type of this
        object to the one given.

        applies only to :class:`_expression.ColumnElement` classes.

        """
        return self

    @property
    def _constructor(self):
        """return the 'constructor' for this ClauseElement.

        This is for the purposes for creating a new object of
        this type.   Usually, its just the element's __class__.
        However, the "Annotated" version of the object overrides
        to return the class of its proxied element.

        """
        return self.__class__

    @HasMemoized.memoized_attribute
    def _cloned_set(self):
        """Return the set consisting all cloned ancestors of this
        ClauseElement.

        Includes this ClauseElement.  This accessor tends to be used for
        FromClause objects to identify 'equivalent' FROM clauses, regardless
        of transformative operations.

        """
        s = util.column_set()
        f: Optional[ClauseElement] = self
        while f is not None:
            s.add(f)
            f = f._is_clone_of
        return s

    def _de_clone(self):
        while self._is_clone_of is not None:
            self = self._is_clone_of
        return self

    @property
    def entity_namespace(self):
        raise AttributeError('This SQL expression has no entity namespace with which to filter from.')

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_is_clone_of', None)
        d.pop('_generate_cache_key', None)
        return d

    def _execute_on_connection(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> Result[Any]:
        if self.supports_execution:
            if TYPE_CHECKING:
                assert isinstance(self, Executable)
            return connection._execute_clauseelement(self, distilled_params, execution_options)
        else:
            raise exc.ObjectNotExecutableError(self)

    def _execute_on_scalar(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> Any:
        """an additional hook for subclasses to provide a different
        implementation for connection.scalar() vs. connection.execute().

        .. versionadded:: 2.0

        """
        return self._execute_on_connection(connection, distilled_params, execution_options).scalar()

    def _get_embedded_bindparams(self) -> Sequence[BindParameter[Any]]:
        """Return the list of :class:`.BindParameter` objects embedded in the
        object.

        This accomplishes the same purpose as ``visitors.traverse()`` or
        similar would provide, however by making use of the cache key
        it takes advantage of memoization of the key to result in fewer
        net method calls, assuming the statement is also going to be
        executed.

        """
        key = self._generate_cache_key()
        if key is None:
            bindparams: List[BindParameter[Any]] = []
            traverse(self, {}, {'bindparam': bindparams.append})
            return bindparams
        else:
            return key.bindparams

    def unique_params(self, __optionaldict: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Self:
        """Return a copy with :func:`_expression.bindparam` elements
        replaced.

        Same functionality as :meth:`_expression.ClauseElement.params`,
        except adds `unique=True`
        to affected bind parameters so that multiple statements can be
        used.

        """
        return self._replace_params(True, __optionaldict, kwargs)

    def params(self, __optionaldict: Optional[Mapping[str, Any]]=None, **kwargs: Any) -> Self:
        """Return a copy with :func:`_expression.bindparam` elements
        replaced.

        Returns a copy of this ClauseElement with
        :func:`_expression.bindparam`
        elements replaced with values taken from the given dictionary::

          >>> clause = column('x') + bindparam('foo')
          >>> print(clause.compile().params)
          {'foo':None}
          >>> print(clause.params({'foo':7}).compile().params)
          {'foo':7}

        """
        return self._replace_params(False, __optionaldict, kwargs)

    def _replace_params(self, unique: bool, optionaldict: Optional[Mapping[str, Any]], kwargs: Dict[str, Any]) -> Self:
        if optionaldict:
            kwargs.update(optionaldict)

        def visit_bindparam(bind: BindParameter[Any]) -> None:
            if bind.key in kwargs:
                bind.value = kwargs[bind.key]
                bind.required = False
            if unique:
                bind._convert_to_unique()
        return cloned_traverse(self, {'maintain_key': True, 'detect_subquery_cols': True}, {'bindparam': visit_bindparam})

    def compare(self, other: ClauseElement, **kw: Any) -> bool:
        """Compare this :class:`_expression.ClauseElement` to
        the given :class:`_expression.ClauseElement`.

        Subclasses should override the default behavior, which is a
        straight identity comparison.

        \\**kw are arguments consumed by subclass ``compare()`` methods and
        may be used to modify the criteria for comparison
        (see :class:`_expression.ColumnElement`).

        """
        return traversals.compare(self, other, **kw)

    def self_group(self, against: Optional[OperatorType]=None) -> ClauseElement:
        """Apply a 'grouping' to this :class:`_expression.ClauseElement`.

        This method is overridden by subclasses to return a "grouping"
        construct, i.e. parenthesis.   In particular it's used by "binary"
        expressions to provide a grouping around themselves when placed into a
        larger expression, as well as by :func:`_expression.select`
        constructs when placed into the FROM clause of another
        :func:`_expression.select`.  (Note that subqueries should be
        normally created using the :meth:`_expression.Select.alias` method,
        as many
        platforms require nested SELECT statements to be named).

        As expressions are composed together, the application of
        :meth:`self_group` is automatic - end-user code should never
        need to use this method directly.  Note that SQLAlchemy's
        clause constructs take operator precedence into account -
        so parenthesis might not be needed, for example, in
        an expression like ``x OR (y AND z)`` - AND takes precedence
        over OR.

        The base :meth:`self_group` method of
        :class:`_expression.ClauseElement`
        just returns self.
        """
        return self

    def _ungroup(self) -> ClauseElement:
        """Return this :class:`_expression.ClauseElement`
        without any groupings.
        """
        return self

    def _compile_w_cache(self, dialect: Dialect, *, compiled_cache: Optional[CompiledCacheType], column_keys: List[str], for_executemany: bool=False, schema_translate_map: Optional[SchemaTranslateMapType]=None, **kw: Any) -> typing_Tuple[Compiled, Optional[Sequence[BindParameter[Any]]], CacheStats]:
        elem_cache_key: Optional[CacheKey]
        if compiled_cache is not None and dialect._supports_statement_cache:
            elem_cache_key = self._generate_cache_key()
        else:
            elem_cache_key = None
        if elem_cache_key is not None:
            if TYPE_CHECKING:
                assert compiled_cache is not None
            cache_key, extracted_params = elem_cache_key
            key = (dialect, cache_key, tuple(column_keys), bool(schema_translate_map), for_executemany)
            compiled_sql = compiled_cache.get(key)
            if compiled_sql is None:
                cache_hit = dialect.CACHE_MISS
                compiled_sql = self._compiler(dialect, cache_key=elem_cache_key, column_keys=column_keys, for_executemany=for_executemany, schema_translate_map=schema_translate_map, **kw)
                compiled_cache[key] = compiled_sql
            else:
                cache_hit = dialect.CACHE_HIT
        else:
            extracted_params = None
            compiled_sql = self._compiler(dialect, cache_key=elem_cache_key, column_keys=column_keys, for_executemany=for_executemany, schema_translate_map=schema_translate_map, **kw)
            if not dialect._supports_statement_cache:
                cache_hit = dialect.NO_DIALECT_SUPPORT
            elif compiled_cache is None:
                cache_hit = dialect.CACHING_DISABLED
            else:
                cache_hit = dialect.NO_CACHE_KEY
        return (compiled_sql, extracted_params, cache_hit)

    def __invert__(self):
        if hasattr(self, 'negation_clause'):
            return self.negation_clause
        else:
            return self._negate()

    def _negate(self) -> ClauseElement:
        grouped = self.self_group(against=operators.inv)
        assert isinstance(grouped, ColumnElement)
        return UnaryExpression(grouped, operator=operators.inv)

    def __bool__(self):
        raise TypeError('Boolean value of this clause is not defined')

    def __repr__(self):
        friendly = self.description
        if friendly is None:
            return object.__repr__(self)
        else:
            return '<%s.%s at 0x%x; %s>' % (self.__module__, self.__class__.__name__, id(self), friendly)