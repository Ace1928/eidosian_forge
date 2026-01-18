import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
class ReturningGuardsTest(fixtures.TablesTest):
    """test that the various 'returning' flags are set appropriately"""
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('t', metadata, Column('id', Integer, primary_key=True, autoincrement=False), Column('data', String(50)))

    @testing.fixture
    def run_stmt(self, connection):
        t = self.tables.t

        def go(stmt, executemany, id_param_name, expect_success):
            stmt = stmt.returning(t.c.id)
            if executemany:
                if not expect_success:
                    with expect_raises_message(exc.StatementError, f'Dialect {connection.dialect.name}\\+{connection.dialect.driver} with current server capabilities does not support .*RETURNING when executemany is used'):
                        result = connection.execute(stmt, [{id_param_name: 1, 'data': 'd1'}, {id_param_name: 2, 'data': 'd2'}, {id_param_name: 3, 'data': 'd3'}])
                else:
                    result = connection.execute(stmt, [{id_param_name: 1, 'data': 'd1'}, {id_param_name: 2, 'data': 'd2'}, {id_param_name: 3, 'data': 'd3'}])
                    eq_(result.all(), [(1,), (2,), (3,)])
            elif not expect_success:
                with expect_raises(exc.DBAPIError):
                    connection.execute(stmt, {id_param_name: 1, 'data': 'd1'})
            else:
                result = connection.execute(stmt, {id_param_name: 1, 'data': 'd1'})
                eq_(result.all(), [(1,)])
        return go

    def test_insert_single(self, connection, run_stmt):
        t = self.tables.t
        stmt = t.insert()
        run_stmt(stmt, False, 'id', connection.dialect.insert_returning)

    def test_insert_many(self, connection, run_stmt):
        t = self.tables.t
        stmt = t.insert()
        run_stmt(stmt, True, 'id', connection.dialect.insert_executemany_returning)

    def test_update_single(self, connection, run_stmt):
        t = self.tables.t
        connection.execute(t.insert(), [{'id': 1, 'data': 'd1'}, {'id': 2, 'data': 'd2'}, {'id': 3, 'data': 'd3'}])
        stmt = t.update().where(t.c.id == bindparam('b_id'))
        run_stmt(stmt, False, 'b_id', connection.dialect.update_returning)

    def test_update_many(self, connection, run_stmt):
        t = self.tables.t
        connection.execute(t.insert(), [{'id': 1, 'data': 'd1'}, {'id': 2, 'data': 'd2'}, {'id': 3, 'data': 'd3'}])
        stmt = t.update().where(t.c.id == bindparam('b_id'))
        run_stmt(stmt, True, 'b_id', connection.dialect.update_executemany_returning)

    def test_delete_single(self, connection, run_stmt):
        t = self.tables.t
        connection.execute(t.insert(), [{'id': 1, 'data': 'd1'}, {'id': 2, 'data': 'd2'}, {'id': 3, 'data': 'd3'}])
        stmt = t.delete().where(t.c.id == bindparam('b_id'))
        run_stmt(stmt, False, 'b_id', connection.dialect.delete_returning)

    def test_delete_many(self, connection, run_stmt):
        t = self.tables.t
        connection.execute(t.insert(), [{'id': 1, 'data': 'd1'}, {'id': 2, 'data': 'd2'}, {'id': 3, 'data': 'd3'}])
        stmt = t.delete().where(t.c.id == bindparam('b_id'))
        run_stmt(stmt, True, 'b_id', connection.dialect.delete_executemany_returning)