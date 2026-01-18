from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import Integer
from ... import select
from ... import testing
from ... import union
def _dont_test_select_from_plain_union(self, connection):
    table = self.tables.some_table
    s1 = select(table).where(table.c.id == 2)
    s2 = select(table).where(table.c.id == 3)
    u1 = union(s1, s2).alias().select()
    with testing.expect_deprecated('The SelectBase.c and SelectBase.columns attributes are deprecated'):
        self._assert_result(connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)])