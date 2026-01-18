from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TypeVar
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql._typing import _TypeEngineArgument
class ARRAY(sqltypes.ARRAY):
    """PostgreSQL ARRAY type.

    The :class:`_postgresql.ARRAY` type is constructed in the same way
    as the core :class:`_types.ARRAY` type; a member type is required, and a
    number of dimensions is recommended if the type is to be used for more
    than one dimension::

        from sqlalchemy.dialects import postgresql

        mytable = Table("mytable", metadata,
                Column("data", postgresql.ARRAY(Integer, dimensions=2))
            )

    The :class:`_postgresql.ARRAY` type provides all operations defined on the
    core :class:`_types.ARRAY` type, including support for "dimensions",
    indexed access, and simple matching such as
    :meth:`.types.ARRAY.Comparator.any` and
    :meth:`.types.ARRAY.Comparator.all`.  :class:`_postgresql.ARRAY`
    class also
    provides PostgreSQL-specific methods for containment operations, including
    :meth:`.postgresql.ARRAY.Comparator.contains`
    :meth:`.postgresql.ARRAY.Comparator.contained_by`, and
    :meth:`.postgresql.ARRAY.Comparator.overlap`, e.g.::

        mytable.c.data.contains([1, 2])

    Indexed access is one-based by default, to match that of PostgreSQL;
    for zero-based indexed access, set
    :paramref:`_postgresql.ARRAY.zero_indexes`.

    Additionally, the :class:`_postgresql.ARRAY`
    type does not work directly in
    conjunction with the :class:`.ENUM` type.  For a workaround, see the
    special type at :ref:`postgresql_array_of_enum`.

    .. container:: topic

        **Detecting Changes in ARRAY columns when using the ORM**

        The :class:`_postgresql.ARRAY` type, when used with the SQLAlchemy ORM,
        does not detect in-place mutations to the array. In order to detect
        these, the :mod:`sqlalchemy.ext.mutable` extension must be used, using
        the :class:`.MutableList` class::

            from sqlalchemy.dialects.postgresql import ARRAY
            from sqlalchemy.ext.mutable import MutableList

            class SomeOrmClass(Base):
                # ...

                data = Column(MutableList.as_mutable(ARRAY(Integer)))

        This extension will allow "in-place" changes such to the array
        such as ``.append()`` to produce events which will be detected by the
        unit of work.  Note that changes to elements **inside** the array,
        including subarrays that are mutated in place, are **not** detected.

        Alternatively, assigning a new array value to an ORM element that
        replaces the old one will always trigger a change event.

    .. seealso::

        :class:`_types.ARRAY` - base array type

        :class:`_postgresql.array` - produces a literal array value.

    """

    def __init__(self, item_type: _TypeEngineArgument[Any], as_tuple: bool=False, dimensions: Optional[int]=None, zero_indexes: bool=False):
        """Construct an ARRAY.

        E.g.::

          Column('myarray', ARRAY(Integer))

        Arguments are:

        :param item_type: The data type of items of this array. Note that
          dimensionality is irrelevant here, so multi-dimensional arrays like
          ``INTEGER[][]``, are constructed as ``ARRAY(Integer)``, not as
          ``ARRAY(ARRAY(Integer))`` or such.

        :param as_tuple=False: Specify whether return results
          should be converted to tuples from lists. DBAPIs such
          as psycopg2 return lists by default. When tuples are
          returned, the results are hashable.

        :param dimensions: if non-None, the ARRAY will assume a fixed
         number of dimensions.  This will cause the DDL emitted for this
         ARRAY to include the exact number of bracket clauses ``[]``,
         and will also optimize the performance of the type overall.
         Note that PG arrays are always implicitly "non-dimensioned",
         meaning they can store any number of dimensions no matter how
         they were declared.

        :param zero_indexes=False: when True, index values will be converted
         between Python zero-based and PostgreSQL one-based indexes, e.g.
         a value of one will be added to all index values before passing
         to the database.

        """
        if isinstance(item_type, ARRAY):
            raise ValueError('Do not nest ARRAY types; ARRAY(basetype) handles multi-dimensional arrays of basetype')
        if isinstance(item_type, type):
            item_type = item_type()
        self.item_type = item_type
        self.as_tuple = as_tuple
        self.dimensions = dimensions
        self.zero_indexes = zero_indexes

    class Comparator(sqltypes.ARRAY.Comparator):
        """Define comparison operations for :class:`_types.ARRAY`.

        Note that these operations are in addition to those provided
        by the base :class:`.types.ARRAY.Comparator` class, including
        :meth:`.types.ARRAY.Comparator.any` and
        :meth:`.types.ARRAY.Comparator.all`.

        """

        def contains(self, other, **kwargs):
            """Boolean expression.  Test if elements are a superset of the
            elements of the argument array expression.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            return self.operate(CONTAINS, other, result_type=sqltypes.Boolean)

        def contained_by(self, other):
            """Boolean expression.  Test if elements are a proper subset of the
            elements of the argument array expression.
            """
            return self.operate(CONTAINED_BY, other, result_type=sqltypes.Boolean)

        def overlap(self, other):
            """Boolean expression.  Test if array has elements in common with
            an argument array expression.
            """
            return self.operate(OVERLAP, other, result_type=sqltypes.Boolean)
    comparator_factory = Comparator

    @property
    def hashable(self):
        return self.as_tuple

    @property
    def python_type(self):
        return list

    def compare_values(self, x, y):
        return x == y

    @util.memoized_property
    def _against_native_enum(self):
        return isinstance(self.item_type, sqltypes.Enum) and self.item_type.native_enum

    def literal_processor(self, dialect):
        item_proc = self.item_type.dialect_impl(dialect).literal_processor(dialect)
        if item_proc is None:
            return None

        def to_str(elements):
            return f'ARRAY[{', '.join(elements)}]'

        def process(value):
            inner = self._apply_item_processor(value, item_proc, self.dimensions, to_str)
            return inner
        return process

    def bind_processor(self, dialect):
        item_proc = self.item_type.dialect_impl(dialect).bind_processor(dialect)

        def process(value):
            if value is None:
                return value
            else:
                return self._apply_item_processor(value, item_proc, self.dimensions, list)
        return process

    def result_processor(self, dialect, coltype):
        item_proc = self.item_type.dialect_impl(dialect).result_processor(dialect, coltype)

        def process(value):
            if value is None:
                return value
            else:
                return self._apply_item_processor(value, item_proc, self.dimensions, tuple if self.as_tuple else list)
        if self._against_native_enum:
            super_rp = process
            pattern = re.compile('^{(.*)}$')

            def handle_raw_string(value):
                inner = pattern.match(value).group(1)
                return _split_enum_values(inner)

            def process(value):
                if value is None:
                    return value
                return super_rp(handle_raw_string(value) if isinstance(value, str) else value)
        return process