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
class PostCompileParamsTest(AssertsExecutionResults, AssertsCompiledSQL, fixtures.TablesTest):
    __backend__ = True
    __requires__ = ('standard_cursor_sql',)

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('x', Integer), Column('y', Integer), Column('z', String(50)))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'x': 1, 'y': 2, 'z': 'z1'}, {'id': 2, 'x': 2, 'y': 3, 'z': 'z2'}, {'id': 3, 'x': 3, 'y': 4, 'z': 'z3'}, {'id': 4, 'x': 4, 'y': 5, 'z': 'z4'}])

    def test_compile(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x == bindparam('q', literal_execute=True))
        self.assert_compile(stmt, 'SELECT some_table.id FROM some_table WHERE some_table.x = __[POSTCOMPILE_q]', {})

    def test_compile_literal_binds(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x == bindparam('q', 10, literal_execute=True))
        self.assert_compile(stmt, 'SELECT some_table.id FROM some_table WHERE some_table.x = 10', {}, literal_binds=True)

    def test_execute(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x == bindparam('q', literal_execute=True))
        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=10))
        asserter.assert_(CursorSQL('SELECT some_table.id \nFROM some_table \nWHERE some_table.x = 10', () if config.db.dialect.positional else {}))

    def test_execute_expanding_plus_literal_execute(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_(bindparam('q', expanding=True, literal_execute=True)))
        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=[5, 6, 7]))
        asserter.assert_(CursorSQL('SELECT some_table.id \nFROM some_table \nWHERE some_table.x IN (5, 6, 7)', () if config.db.dialect.positional else {}))

    @testing.requires.tuple_in
    def test_execute_tuple_expanding_plus_literal_execute(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.y).in_(bindparam('q', expanding=True, literal_execute=True)))
        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=[(5, 10), (12, 18)]))
        asserter.assert_(CursorSQL('SELECT some_table.id \nFROM some_table \nWHERE (some_table.x, some_table.y) IN (%s(5, 10), (12, 18))' % ('VALUES ' if config.db.dialect.tuple_in_values else ''), () if config.db.dialect.positional else {}))

    @testing.requires.tuple_in
    def test_execute_tuple_expanding_plus_literal_heterogeneous_execute(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(tuple_(table.c.x, table.c.z).in_(bindparam('q', expanding=True, literal_execute=True)))
        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=[(5, 'z1'), (12, 'z3')]))
        asserter.assert_(CursorSQL("SELECT some_table.id \nFROM some_table \nWHERE (some_table.x, some_table.z) IN (%s(5, 'z1'), (12, 'z3'))" % ('VALUES ' if config.db.dialect.tuple_in_values else ''), () if config.db.dialect.positional else {}))