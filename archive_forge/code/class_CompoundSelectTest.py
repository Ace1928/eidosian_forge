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
class CompoundSelectTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('x', Integer), Column('y', Integer))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'x': 1, 'y': 2}, {'id': 2, 'x': 2, 'y': 3}, {'id': 3, 'x': 3, 'y': 4}, {'id': 4, 'x': 4, 'y': 5}])

    def _assert_result(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.execute(select, params).fetchall(), result)

    def test_plain_union(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2)
        s2 = select(table).where(table.c.id == 3)
        u1 = union(s1, s2)
        self._assert_result(u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)])

    def test_select_from_plain_union(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2)
        s2 = select(table).where(table.c.id == 3)
        u1 = union(s1, s2).alias().select()
        self._assert_result(u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)])

    @testing.requires.order_by_col_from_union
    @testing.requires.parens_in_union_contained_select_w_limit_offset
    def test_limit_offset_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).limit(1).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).limit(1).order_by(table.c.id)
        u1 = union(s1, s2).limit(2)
        self._assert_result(u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)])

    @testing.requires.parens_in_union_contained_select_wo_limit_offset
    def test_order_by_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).order_by(table.c.id)
        u1 = union(s1, s2).limit(2)
        self._assert_result(u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)])

    def test_distinct_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).distinct()
        s2 = select(table).where(table.c.id == 3).distinct()
        u1 = union(s1, s2).limit(2)
        self._assert_result(u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)])

    @testing.requires.parens_in_union_contained_select_w_limit_offset
    def test_limit_offset_in_unions_from_alias(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).limit(1).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).limit(1).order_by(table.c.id)
        u1 = union(s1, s2).alias()
        self._assert_result(u1.select().limit(2).order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)])

    def test_limit_offset_aliased_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).limit(1).order_by(table.c.id).alias().select()
        s2 = select(table).where(table.c.id == 3).limit(1).order_by(table.c.id).alias().select()
        u1 = union(s1, s2).limit(2)
        self._assert_result(u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)])