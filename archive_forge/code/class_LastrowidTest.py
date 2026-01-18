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
class LastrowidTest(fixtures.TablesTest):
    run_deletes = 'each'
    __backend__ = True
    __requires__ = ('implements_get_lastrowid', 'autoincrement_insert')

    @classmethod
    def define_tables(cls, metadata):
        Table('autoinc_pk', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('data', String(50)), implicit_returning=False)
        Table('manual_pk', metadata, Column('id', Integer, primary_key=True, autoincrement=False), Column('data', String(50)), implicit_returning=False)

    def _assert_round_trip(self, table, conn):
        row = conn.execute(table.select()).first()
        eq_(row, (conn.dialect.default_sequence_base, 'some data'))

    def test_autoincrement_on_insert(self, connection):
        connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
        self._assert_round_trip(self.tables.autoinc_pk, connection)

    def test_last_inserted_id(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
        pk = connection.scalar(select(self.tables.autoinc_pk.c.id))
        eq_(r.inserted_primary_key, (pk,))

    @requirements.dbapi_lastrowid
    def test_native_lastrowid_autoinc(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
        lastrowid = r.lastrowid
        pk = connection.scalar(select(self.tables.autoinc_pk.c.id))
        eq_(lastrowid, pk)