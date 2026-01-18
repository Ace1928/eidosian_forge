from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
class InsertBehaviorTest(fixtures.TablesTest):
    run_deletes = 'each'
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('autoinc_pk', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('data', String(50)))
        Table('manual_pk', metadata, Column('id', Integer, primary_key=True, autoincrement=False), Column('data', String(50)))
        Table('no_implicit_returning', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('data', String(50)), implicit_returning=False)
        Table('includes_defaults', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('data', String(50)), Column('x', Integer, default=5), Column('y', Integer, default=literal_column('2', type_=Integer) + literal(2)))

    @testing.variation('style', ['plain', 'return_defaults'])
    @testing.variation('executemany', [True, False])
    def test_no_results_for_non_returning_insert(self, connection, style, executemany):
        """test another INSERT issue found during #10453"""
        table = self.tables.no_implicit_returning
        stmt = table.insert()
        if style.return_defaults:
            stmt = stmt.return_defaults()
        if executemany:
            data = [{'data': 'd1'}, {'data': 'd2'}, {'data': 'd3'}, {'data': 'd4'}, {'data': 'd5'}]
        else:
            data = {'data': 'd1'}
        r = connection.execute(stmt, data)
        assert not r.returns_rows

    @requirements.autoincrement_insert
    def test_autoclose_on_insert(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
        assert r._soft_closed
        assert not r.closed
        assert r.is_insert
        assert not r.returns_rows or r.fetchone() is None

    @requirements.insert_returning
    def test_autoclose_on_insert_implicit_returning(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert().return_defaults(), dict(data='some data'))
        assert r._soft_closed
        assert not r.closed
        assert r.is_insert
        assert r.returns_rows
        eq_(r.fetchone(), None)
        eq_(r.keys(), ['id'])

    @requirements.empty_inserts
    def test_empty_insert(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert())
        assert r._soft_closed
        assert not r.closed
        r = connection.execute(self.tables.autoinc_pk.select().where(self.tables.autoinc_pk.c.id != None))
        eq_(len(r.all()), 1)

    @requirements.empty_inserts_executemany
    def test_empty_insert_multiple(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert(), [{}, {}, {}])
        assert r._soft_closed
        assert not r.closed
        r = connection.execute(self.tables.autoinc_pk.select().where(self.tables.autoinc_pk.c.id != None))
        eq_(len(r.all()), 3)

    @requirements.insert_from_select
    def test_insert_from_select_autoinc(self, connection):
        src_table = self.tables.manual_pk
        dest_table = self.tables.autoinc_pk
        connection.execute(src_table.insert(), [dict(id=1, data='data1'), dict(id=2, data='data2'), dict(id=3, data='data3')])
        result = connection.execute(dest_table.insert().from_select(('data',), select(src_table.c.data).where(src_table.c.data.in_(['data2', 'data3']))))
        eq_(result.inserted_primary_key, (None,))
        result = connection.execute(select(dest_table.c.data).order_by(dest_table.c.data))
        eq_(result.fetchall(), [('data2',), ('data3',)])

    @requirements.insert_from_select
    def test_insert_from_select_autoinc_no_rows(self, connection):
        src_table = self.tables.manual_pk
        dest_table = self.tables.autoinc_pk
        result = connection.execute(dest_table.insert().from_select(('data',), select(src_table.c.data).where(src_table.c.data.in_(['data2', 'data3']))))
        eq_(result.inserted_primary_key, (None,))
        result = connection.execute(select(dest_table.c.data).order_by(dest_table.c.data))
        eq_(result.fetchall(), [])

    @requirements.insert_from_select
    def test_insert_from_select(self, connection):
        table = self.tables.manual_pk
        connection.execute(table.insert(), [dict(id=1, data='data1'), dict(id=2, data='data2'), dict(id=3, data='data3')])
        connection.execute(table.insert().inline().from_select(('id', 'data'), select(table.c.id + 5, table.c.data).where(table.c.data.in_(['data2', 'data3']))))
        eq_(connection.execute(select(table.c.data).order_by(table.c.data)).fetchall(), [('data1',), ('data2',), ('data2',), ('data3',), ('data3',)])

    @requirements.insert_from_select
    def test_insert_from_select_with_defaults(self, connection):
        table = self.tables.includes_defaults
        connection.execute(table.insert(), [dict(id=1, data='data1'), dict(id=2, data='data2'), dict(id=3, data='data3')])
        connection.execute(table.insert().inline().from_select(('id', 'data'), select(table.c.id + 5, table.c.data).where(table.c.data.in_(['data2', 'data3']))))
        eq_(connection.execute(select(table).order_by(table.c.data, table.c.id)).fetchall(), [(1, 'data1', 5, 4), (2, 'data2', 5, 4), (7, 'data2', 5, 4), (3, 'data3', 5, 4), (8, 'data3', 5, 4)])