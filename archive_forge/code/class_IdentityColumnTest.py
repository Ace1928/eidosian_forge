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
class IdentityColumnTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ('identity_columns',)
    run_inserts = 'once'
    run_deletes = 'once'

    @classmethod
    def define_tables(cls, metadata):
        Table('tbl_a', metadata, Column('id', Integer, Identity(always=True, start=42, nominvalue=True, nomaxvalue=True), primary_key=True), Column('desc', String(100)))
        Table('tbl_b', metadata, Column('id', Integer, Identity(increment=-5, start=0, minvalue=-1000, maxvalue=0), primary_key=True), Column('desc', String(100)))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.tbl_a.insert(), [{'desc': 'a'}, {'desc': 'b'}])
        connection.execute(cls.tables.tbl_b.insert(), [{'desc': 'a'}, {'desc': 'b'}])
        connection.execute(cls.tables.tbl_b.insert(), [{'id': 42, 'desc': 'c'}])

    def test_select_all(self, connection):
        res = connection.execute(select(text('*')).select_from(self.tables.tbl_a).order_by(self.tables.tbl_a.c.id)).fetchall()
        eq_(res, [(42, 'a'), (43, 'b')])
        res = connection.execute(select(text('*')).select_from(self.tables.tbl_b).order_by(self.tables.tbl_b.c.id)).fetchall()
        eq_(res, [(-5, 'b'), (0, 'a'), (42, 'c')])

    def test_select_columns(self, connection):
        res = connection.execute(select(self.tables.tbl_a.c.id).order_by(self.tables.tbl_a.c.id)).fetchall()
        eq_(res, [(42,), (43,)])

    @testing.requires.identity_columns_standard
    def test_insert_always_error(self, connection):

        def fn():
            connection.execute(self.tables.tbl_a.insert(), [{'id': 200, 'desc': 'a'}])
        assert_raises((DatabaseError, ProgrammingError), fn)