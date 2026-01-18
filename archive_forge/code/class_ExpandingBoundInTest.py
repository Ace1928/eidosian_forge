import collections.abc as collections_abc
import itertools
from .. import AssertsCompiledSQL
from .. import AssertsExecutionResults
from .. import config
from .. import fixtures
from ..assertions import assert_raises
from ..assertions import eq_
from ..assertions import in_
from ..assertsql import CursorSQL
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import case
from ... import column
from ... import Computed
from ... import exists
from ... import false
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import null
from ... import select
from ... import String
from ... import table
from ... import testing
from ... import text
from ... import true
from ... import tuple_
from ... import TupleType
from ... import union
from ... import values
from ...exc import DatabaseError
from ...exc import ProgrammingError
class ExpandingBoundInTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('x', Integer), Column('y', Integer), Column('z', String(50)))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'x': 1, 'y': 2, 'z': 'z1'}, {'id': 2, 'x': 2, 'y': 3, 'z': 'z2'}, {'id': 3, 'x': 3, 'y': 4, 'z': 'z3'}, {'id': 4, 'x': 4, 'y': 5, 'z': 'z4'}])

    def _assert_result(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.execute(select, params).fetchall(), result)

    def test_multiple_empty_sets_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_(bindparam('q'))).where(table.c.y.in_(bindparam('p'))).order_by(table.c.id)
        self._assert_result(stmt, [], params={'q': [], 'p': []})

    def test_multiple_empty_sets_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_([])).where(table.c.y.in_([])).order_by(table.c.id)
        self._assert_result(stmt, [])

    @testing.requires.tuple_in_w_empty
    def test_empty_heterogeneous_tuples_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.z).in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [], params={'q': []})

    @testing.requires.tuple_in_w_empty
    def test_empty_heterogeneous_tuples_direct(self):
        table = self.tables.some_table

        def go(val, expected):
            stmt = select(table.c.id).where(tuple_(table.c.x, table.c.z).in_(val)).order_by(table.c.id)
            self._assert_result(stmt, expected)
        go([], [])
        go([(2, 'z2'), (3, 'z3'), (4, 'z4')], [(2,), (3,), (4,)])
        go([], [])

    @testing.requires.tuple_in_w_empty
    def test_empty_homogeneous_tuples_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.y).in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [], params={'q': []})

    @testing.requires.tuple_in_w_empty
    def test_empty_homogeneous_tuples_direct(self):
        table = self.tables.some_table

        def go(val, expected):
            stmt = select(table.c.id).where(tuple_(table.c.x, table.c.y).in_(val)).order_by(table.c.id)
            self._assert_result(stmt, expected)
        go([], [])
        go([(1, 2), (2, 3), (3, 4)], [(1,), (2,), (3,)])
        go([], [])

    def test_bound_in_scalar_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [2, 3, 4]})

    def test_bound_in_scalar_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_([2, 3, 4])).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,), (4,)])

    def test_nonempty_in_plus_empty_notin(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_([2, 3])).where(table.c.id.not_in([])).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,)])

    def test_empty_in_plus_notempty_notin(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_([])).where(table.c.id.not_in([2, 3])).order_by(table.c.id)
        self._assert_result(stmt, [])

    def test_typed_str_in(self):
        """test related to #7292.

        as a type is given to the bound param, there is no ambiguity
        to the type of element.

        """
        stmt = text('select id FROM some_table WHERE z IN :q ORDER BY id').bindparams(bindparam('q', type_=String, expanding=True))
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': ['z2', 'z3', 'z4']})

    def test_untyped_str_in(self):
        """test related to #7292.

        for untyped expression, we look at the types of elements.
        Test for Sequence to detect tuple in.  but not strings or bytes!
        as always....

        """
        stmt = text('select id FROM some_table WHERE z IN :q ORDER BY id').bindparams(bindparam('q', expanding=True))
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': ['z2', 'z3', 'z4']})

    @testing.requires.tuple_in
    def test_bound_in_two_tuple_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.y).in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [(2, 3), (3, 4), (4, 5)]})

    @testing.requires.tuple_in
    def test_bound_in_two_tuple_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.y).in_([(2, 3), (3, 4), (4, 5)])).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,), (4,)])

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.z).in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [(2, 'z2'), (3, 'z3'), (4, 'z4')]})

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.z).in_([(2, 'z2'), (3, 'z3'), (4, 'z4')])).order_by(table.c.id)
        self._assert_result(stmt, [(2,), (3,), (4,)])

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_text_bindparam(self):
        stmt = text('select id FROM some_table WHERE (x, z) IN :q ORDER BY id').bindparams(bindparam('q', expanding=True))
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [(2, 'z2'), (3, 'z3'), (4, 'z4')]})

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_typed_bindparam_non_tuple(self):

        class LikeATuple(collections_abc.Sequence):

            def __init__(self, *data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)
        stmt = text('select id FROM some_table WHERE (x, z) IN :q ORDER BY id').bindparams(bindparam('q', type_=TupleType(Integer(), String()), expanding=True))
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [LikeATuple(2, 'z2'), LikeATuple(3, 'z3'), LikeATuple(4, 'z4')]})

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_text_bindparam_non_tuple(self):

        class LikeATuple(collections_abc.Sequence):

            def __init__(self, *data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)
        stmt = text('select id FROM some_table WHERE (x, z) IN :q ORDER BY id').bindparams(bindparam('q', expanding=True))
        self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [LikeATuple(2, 'z2'), LikeATuple(3, 'z3'), LikeATuple(4, 'z4')]})

    def test_empty_set_against_integer_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [], params={'q': []})

    def test_empty_set_against_integer_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_([])).order_by(table.c.id)
        self._assert_result(stmt, [])

    def test_empty_set_against_integer_negation_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.not_in(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)], params={'q': []})

    def test_empty_set_against_integer_negation_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.not_in([])).order_by(table.c.id)
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)])

    def test_empty_set_against_string_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.z.in_(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [], params={'q': []})

    def test_empty_set_against_string_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.z.in_([])).order_by(table.c.id)
        self._assert_result(stmt, [])

    def test_empty_set_against_string_negation_bindparam(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.z.not_in(bindparam('q'))).order_by(table.c.id)
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)], params={'q': []})

    def test_empty_set_against_string_negation_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.z.not_in([])).order_by(table.c.id)
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)])

    def test_null_in_empty_set_is_false_bindparam(self, connection):
        stmt = select(case((null().in_(bindparam('foo', value=())), true()), else_=false()))
        in_(connection.execute(stmt).fetchone()[0], (False, 0))

    def test_null_in_empty_set_is_false_direct(self, connection):
        stmt = select(case((null().in_([]), true()), else_=false()))
        in_(connection.execute(stmt).fetchone()[0], (False, 0))