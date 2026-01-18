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
class IsOrIsNotDistinctFromTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ('supports_is_distinct_from',)

    @classmethod
    def define_tables(cls, metadata):
        Table('is_distinct_test', metadata, Column('id', Integer, primary_key=True), Column('col_a', Integer, nullable=True), Column('col_b', Integer, nullable=True))

    @testing.combinations(('both_int_different', 0, 1, 1), ('both_int_same', 1, 1, 0), ('one_null_first', None, 1, 1), ('one_null_second', 0, None, 1), ('both_null', None, None, 0), id_='iaaa', argnames='col_a_value, col_b_value, expected_row_count_for_is')
    def test_is_or_is_not_distinct_from(self, col_a_value, col_b_value, expected_row_count_for_is, connection):
        tbl = self.tables.is_distinct_test
        connection.execute(tbl.insert(), [{'id': 1, 'col_a': col_a_value, 'col_b': col_b_value}])
        result = connection.execute(tbl.select().where(tbl.c.col_a.is_distinct_from(tbl.c.col_b))).fetchall()
        eq_(len(result), expected_row_count_for_is)
        expected_row_count_for_is_not = 1 if expected_row_count_for_is == 0 else 0
        result = connection.execute(tbl.select().where(tbl.c.col_a.is_not_distinct_from(tbl.c.col_b))).fetchall()
        eq_(len(result), expected_row_count_for_is_not)