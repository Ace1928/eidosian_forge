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
class ValuesExpressionTest(fixtures.TestBase):
    __requires__ = ('table_value_constructor',)
    __backend__ = True

    def test_tuples(self, connection):
        value_expr = values(column('id', Integer), column('name', String), name='my_values').data([(1, 'name1'), (2, 'name2'), (3, 'name3')])
        eq_(connection.execute(select(value_expr)).all(), [(1, 'name1'), (2, 'name2'), (3, 'name3')])