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
class SameNamedSchemaTableTest(fixtures.TablesTest):
    """tests for #7471"""
    __backend__ = True
    __requires__ = ('schemas',)

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), schema=config.test_schema)
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('some_table_id', Integer, nullable=False))

    @classmethod
    def insert_data(cls, connection):
        some_table, some_table_schema = cls.tables('some_table', '%s.some_table' % config.test_schema)
        connection.execute(some_table_schema.insert(), {'id': 1})
        connection.execute(some_table.insert(), {'id': 1, 'some_table_id': 1})

    def test_simple_join_both_tables(self, connection):
        some_table, some_table_schema = self.tables('some_table', '%s.some_table' % config.test_schema)
        eq_(connection.execute(select(some_table, some_table_schema).join_from(some_table, some_table_schema, some_table.c.some_table_id == some_table_schema.c.id)).first(), (1, 1, 1))

    def test_simple_join_whereclause_only(self, connection):
        some_table, some_table_schema = self.tables('some_table', '%s.some_table' % config.test_schema)
        eq_(connection.execute(select(some_table).join_from(some_table, some_table_schema, some_table.c.some_table_id == some_table_schema.c.id).where(some_table.c.id == 1)).first(), (1, 1))

    def test_subquery(self, connection):
        some_table, some_table_schema = self.tables('some_table', '%s.some_table' % config.test_schema)
        subq = select(some_table).join_from(some_table, some_table_schema, some_table.c.some_table_id == some_table_schema.c.id).where(some_table.c.id == 1).subquery()
        eq_(connection.execute(select(some_table, subq.c.id).join_from(some_table, subq, some_table.c.some_table_id == subq.c.id).where(some_table.c.id == 1)).first(), (1, 1, 1))