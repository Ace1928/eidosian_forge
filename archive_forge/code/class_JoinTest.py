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
class JoinTest(fixtures.TablesTest):
    __backend__ = True

    def _assert_result(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.execute(select, params).fetchall(), result)

    @classmethod
    def define_tables(cls, metadata):
        Table('a', metadata, Column('id', Integer, primary_key=True))
        Table('b', metadata, Column('id', Integer, primary_key=True), Column('a_id', ForeignKey('a.id'), nullable=False))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.a.insert(), [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}])
        connection.execute(cls.tables.b.insert(), [{'id': 1, 'a_id': 1}, {'id': 2, 'a_id': 1}, {'id': 4, 'a_id': 2}, {'id': 5, 'a_id': 3}])

    def test_inner_join_fk(self):
        a, b = self.tables('a', 'b')
        stmt = select(a, b).select_from(a.join(b)).order_by(a.c.id, b.c.id)
        self._assert_result(stmt, [(1, 1, 1), (1, 2, 1), (2, 4, 2), (3, 5, 3)])

    def test_inner_join_true(self):
        a, b = self.tables('a', 'b')
        stmt = select(a, b).select_from(a.join(b, true())).order_by(a.c.id, b.c.id)
        self._assert_result(stmt, [(a, b, c) for (a,), (b, c) in itertools.product([(1,), (2,), (3,), (4,), (5,)], [(1, 1), (2, 1), (4, 2), (5, 3)])])

    def test_inner_join_false(self):
        a, b = self.tables('a', 'b')
        stmt = select(a, b).select_from(a.join(b, false())).order_by(a.c.id, b.c.id)
        self._assert_result(stmt, [])

    def test_outer_join_false(self):
        a, b = self.tables('a', 'b')
        stmt = select(a, b).select_from(a.outerjoin(b, false())).order_by(a.c.id, b.c.id)
        self._assert_result(stmt, [(1, None, None), (2, None, None), (3, None, None), (4, None, None), (5, None, None)])

    def test_outer_join_fk(self):
        a, b = self.tables('a', 'b')
        stmt = select(a, b).select_from(a.join(b)).order_by(a.c.id, b.c.id)
        self._assert_result(stmt, [(1, 1, 1), (1, 2, 1), (2, 4, 2), (3, 5, 3)])