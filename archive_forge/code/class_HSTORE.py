import re
from .array import ARRAY
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import GETITEM
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from ... import types as sqltypes
from ...sql import functions as sqlfunc
class HSTORE(sqltypes.Indexable, sqltypes.Concatenable, sqltypes.TypeEngine):
    """Represent the PostgreSQL HSTORE type.

    The :class:`.HSTORE` type stores dictionaries containing strings, e.g.::

        data_table = Table('data_table', metadata,
            Column('id', Integer, primary_key=True),
            Column('data', HSTORE)
        )

        with engine.connect() as conn:
            conn.execute(
                data_table.insert(),
                data = {"key1": "value1", "key2": "value2"}
            )

    :class:`.HSTORE` provides for a wide range of operations, including:

    * Index operations::

        data_table.c.data['some key'] == 'some value'

    * Containment operations::

        data_table.c.data.has_key('some key')

        data_table.c.data.has_all(['one', 'two', 'three'])

    * Concatenation::

        data_table.c.data + {"k1": "v1"}

    For a full list of special methods see
    :class:`.HSTORE.comparator_factory`.

    .. container:: topic

        **Detecting Changes in HSTORE columns when using the ORM**

        For usage with the SQLAlchemy ORM, it may be desirable to combine the
        usage of :class:`.HSTORE` with :class:`.MutableDict` dictionary now
        part of the :mod:`sqlalchemy.ext.mutable` extension. This extension
        will allow "in-place" changes to the dictionary, e.g. addition of new
        keys or replacement/removal of existing keys to/from the current
        dictionary, to produce events which will be detected by the unit of
        work::

            from sqlalchemy.ext.mutable import MutableDict

            class MyClass(Base):
                __tablename__ = 'data_table'

                id = Column(Integer, primary_key=True)
                data = Column(MutableDict.as_mutable(HSTORE))

            my_object = session.query(MyClass).one()

            # in-place mutation, requires Mutable extension
            # in order for the ORM to detect
            my_object.data['some_key'] = 'some value'

            session.commit()

        When the :mod:`sqlalchemy.ext.mutable` extension is not used, the ORM
        will not be alerted to any changes to the contents of an existing
        dictionary, unless that dictionary value is re-assigned to the
        HSTORE-attribute itself, thus generating a change event.

    .. seealso::

        :class:`.hstore` - render the PostgreSQL ``hstore()`` function.


    """
    __visit_name__ = 'HSTORE'
    hashable = False
    text_type = sqltypes.Text()

    def __init__(self, text_type=None):
        """Construct a new :class:`.HSTORE`.

        :param text_type: the type that should be used for indexed values.
         Defaults to :class:`_types.Text`.

        """
        if text_type is not None:
            self.text_type = text_type

    class Comparator(sqltypes.Indexable.Comparator, sqltypes.Concatenable.Comparator):
        """Define comparison operations for :class:`.HSTORE`."""

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

        def _setup_getitem(self, index):
            return (GETITEM, index, self.type.text_type)

        def defined(self, key):
            """Boolean expression.  Test for presence of a non-NULL value for
            the key.  Note that the key may be a SQLA expression.
            """
            return _HStoreDefinedFunction(self.expr, key)

        def delete(self, key):
            """HStore expression.  Returns the contents of this hstore with the
            given key deleted.  Note that the key may be a SQLA expression.
            """
            if isinstance(key, dict):
                key = _serialize_hstore(key)
            return _HStoreDeleteFunction(self.expr, key)

        def slice(self, array):
            """HStore expression.  Returns a subset of an hstore defined by
            array of keys.
            """
            return _HStoreSliceFunction(self.expr, array)

        def keys(self):
            """Text array expression.  Returns array of keys."""
            return _HStoreKeysFunction(self.expr)

        def vals(self):
            """Text array expression.  Returns array of values."""
            return _HStoreValsFunction(self.expr)

        def array(self):
            """Text array expression.  Returns array of alternating keys and
            values.
            """
            return _HStoreArrayFunction(self.expr)

        def matrix(self):
            """Text array expression.  Returns array of [key, value] pairs."""
            return _HStoreMatrixFunction(self.expr)
    comparator_factory = Comparator

    def bind_processor(self, dialect):

        def process(value):
            if isinstance(value, dict):
                return _serialize_hstore(value)
            else:
                return value
        return process

    def result_processor(self, dialect, coltype):

        def process(value):
            if value is not None:
                return _parse_hstore(value)
            else:
                return value
        return process