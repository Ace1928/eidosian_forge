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
class IdentityAutoincrementTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ('autoincrement_without_sequence',)

    @classmethod
    def define_tables(cls, metadata):
        Table('tbl', metadata, Column('id', Integer, Identity(), primary_key=True, autoincrement=True), Column('desc', String(100)))

    def test_autoincrement_with_identity(self, connection):
        res = connection.execute(self.tables.tbl.insert(), {'desc': 'row'})
        res = connection.execute(self.tables.tbl.select()).first()
        eq_(res, (1, 'row'))