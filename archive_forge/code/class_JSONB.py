from .array import ARRAY
from .array import array as _pg_array
from .operators import ASTEXT
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import DELETE_PATH
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from .operators import JSONPATH_ASTEXT
from .operators import PATH_EXISTS
from .operators import PATH_MATCH
from ... import types as sqltypes
from ...sql import cast
class JSONB(JSON):
    """Represent the PostgreSQL JSONB type.

    The :class:`_postgresql.JSONB` type stores arbitrary JSONB format data,
    e.g.::

        data_table = Table('data_table', metadata,
            Column('id', Integer, primary_key=True),
            Column('data', JSONB)
        )

        with engine.connect() as conn:
            conn.execute(
                data_table.insert(),
                data = {"key1": "value1", "key2": "value2"}
            )

    The :class:`_postgresql.JSONB` type includes all operations provided by
    :class:`_types.JSON`, including the same behaviors for indexing
    operations.
    It also adds additional operators specific to JSONB, including
    :meth:`.JSONB.Comparator.has_key`, :meth:`.JSONB.Comparator.has_all`,
    :meth:`.JSONB.Comparator.has_any`, :meth:`.JSONB.Comparator.contains`,
    :meth:`.JSONB.Comparator.contained_by`,
    :meth:`.JSONB.Comparator.delete_path`,
    :meth:`.JSONB.Comparator.path_exists` and
    :meth:`.JSONB.Comparator.path_match`.

    Like the :class:`_types.JSON` type, the :class:`_postgresql.JSONB`
    type does not detect
    in-place changes when used with the ORM, unless the
    :mod:`sqlalchemy.ext.mutable` extension is used.

    Custom serializers and deserializers
    are shared with the :class:`_types.JSON` class,
    using the ``json_serializer``
    and ``json_deserializer`` keyword arguments.  These must be specified
    at the dialect level using :func:`_sa.create_engine`.  When using
    psycopg2, the serializers are associated with the jsonb type using
    ``psycopg2.extras.register_default_jsonb`` on a per-connection basis,
    in the same way that ``psycopg2.extras.register_default_json`` is used
    to register these handlers with the json type.

    .. seealso::

        :class:`_types.JSON`

    """
    __visit_name__ = 'JSONB'

    class Comparator(JSON.Comparator):
        """Define comparison operations for :class:`_types.JSON`."""

        def has_key(self, other):
            """Boolean expression.  Test for presence of a key.  Note that the
            key may be a SQLA expression.
            """
            return self.operate(HAS_KEY, other, result_type=sqltypes.Boolean)

        def has_all(self, other):
            """Boolean expression.  Test for presence of all keys in jsonb"""
            return self.operate(HAS_ALL, other, result_type=sqltypes.Boolean)

        def has_any(self, other):
            """Boolean expression.  Test for presence of any key in jsonb"""
            return self.operate(HAS_ANY, other, result_type=sqltypes.Boolean)

        def contains(self, other, **kwargs):
            """Boolean expression.  Test if keys (or array) are a superset
            of/contained the keys of the argument jsonb expression.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            return self.operate(CONTAINS, other, result_type=sqltypes.Boolean)

        def contained_by(self, other):
            """Boolean expression.  Test if keys are a proper subset of the
            keys of the argument jsonb expression.
            """
            return self.operate(CONTAINED_BY, other, result_type=sqltypes.Boolean)

        def delete_path(self, array):
            """JSONB expression. Deletes field or array element specified in
            the argument array.

            The input may be a list of strings that will be coerced to an
            ``ARRAY`` or an instance of :meth:`_postgres.array`.

            .. versionadded:: 2.0
            """
            if not isinstance(array, _pg_array):
                array = _pg_array(array)
            right_side = cast(array, ARRAY(sqltypes.TEXT))
            return self.operate(DELETE_PATH, right_side, result_type=JSONB)

        def path_exists(self, other):
            """Boolean expression. Test for presence of item given by the
            argument JSONPath expression.

            .. versionadded:: 2.0
            """
            return self.operate(PATH_EXISTS, other, result_type=sqltypes.Boolean)

        def path_match(self, other):
            """Boolean expression. Test if JSONPath predicate given by the
            argument JSONPath expression matches.

            Only the first item of the result is taken into account.

            .. versionadded:: 2.0
            """
            return self.operate(PATH_MATCH, other, result_type=sqltypes.Boolean)
    comparator_factory = Comparator