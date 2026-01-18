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
class ComputedColumnTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ('computed_columns',)

    @classmethod
    def define_tables(cls, metadata):
        Table('square', metadata, Column('id', Integer, primary_key=True), Column('side', Integer), Column('area', Integer, Computed('side * side')), Column('perimeter', Integer, Computed('4 * side')))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.square.insert(), [{'id': 1, 'side': 10}, {'id': 10, 'side': 42}])

    def test_select_all(self):
        with config.db.connect() as conn:
            res = conn.execute(select(text('*')).select_from(self.tables.square).order_by(self.tables.square.c.id)).fetchall()
            eq_(res, [(1, 10, 100, 40), (10, 42, 1764, 168)])

    def test_select_columns(self):
        with config.db.connect() as conn:
            res = conn.execute(select(self.tables.square.c.area, self.tables.square.c.perimeter).select_from(self.tables.square).order_by(self.tables.square.c.id)).fetchall()
            eq_(res, [(100, 40), (1764, 168)])