import datetime
from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text
class RowFetchTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('plain_pk', metadata, Column('id', Integer, primary_key=True), Column('data', String(50)))
        Table('has_dates', metadata, Column('id', Integer, primary_key=True), Column('today', DateTime))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.plain_pk.insert(), [{'id': 1, 'data': 'd1'}, {'id': 2, 'data': 'd2'}, {'id': 3, 'data': 'd3'}])
        connection.execute(cls.tables.has_dates.insert(), [{'id': 1, 'today': datetime.datetime(2006, 5, 12, 12, 0, 0)}])

    def test_via_attr(self, connection):
        row = connection.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()
        eq_(row.id, 1)
        eq_(row.data, 'd1')

    def test_via_string(self, connection):
        row = connection.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()
        eq_(row._mapping['id'], 1)
        eq_(row._mapping['data'], 'd1')

    def test_via_int(self, connection):
        row = connection.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()
        eq_(row[0], 1)
        eq_(row[1], 'd1')

    def test_via_col_object(self, connection):
        row = connection.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()
        eq_(row._mapping[self.tables.plain_pk.c.id], 1)
        eq_(row._mapping[self.tables.plain_pk.c.data], 'd1')

    @requirements.duplicate_names_in_cursor_description
    def test_row_with_dupe_names(self, connection):
        result = connection.execute(select(self.tables.plain_pk.c.data, self.tables.plain_pk.c.data.label('data')).order_by(self.tables.plain_pk.c.id))
        row = result.first()
        eq_(result.keys(), ['data', 'data'])
        eq_(row, ('d1', 'd1'))

    def test_row_w_scalar_select(self, connection):
        """test that a scalar select as a column is returned as such
        and that type conversion works OK.

        (this is half a SQLAlchemy Core test and half to catch database
        backends that may have unusual behavior with scalar selects.)

        """
        datetable = self.tables.has_dates
        s = select(datetable.alias('x').c.today).scalar_subquery()
        s2 = select(datetable.c.id, s.label('somelabel'))
        row = connection.execute(s2).first()
        eq_(row.somelabel, datetime.datetime(2006, 5, 12, 12, 0, 0))