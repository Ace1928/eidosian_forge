import abc
import netaddr
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_utils import timeutils
from oslo_utils import uuidutils
import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import sqlalchemytypes
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class CIDRTestCase(SqlAlchemyTypesBaseTestCase):

    def _get_test_table(self, meta):
        return sa.Table('fakecidrmodels', meta, sa.Column('id', sa.String(36), primary_key=True, nullable=False), sa.Column('cidr', sqlalchemytypes.CIDR))

    def _get_one(self, value):
        row_select = self.test_table.select().where(self.test_table.c.cidr == value)
        with self.engine.connect() as conn, conn.begin():
            return conn.execute(row_select).first()

    def _update_row(self, key, cidr):
        row_update = self.test_table.update().values(cidr=cidr).where(self.test_table.c.cidr == key)
        with self.engine.connect() as conn, conn.begin():
            conn.execute(row_update)

    def test_crud(self):
        cidrs = ['10.0.0.0/24', '10.123.250.9/32', '2001:db8::/42', 'fe80::21e:67ff:fed0:56f0/64']
        for cidr_str in cidrs:
            cidr = netaddr.IPNetwork(cidr_str)
            self._add_row(id=uuidutils.generate_uuid(), cidr=cidr)
            obj = self._get_one(cidr)
            self.assertEqual(cidr, obj.cidr)
            random_cidr = netaddr.IPNetwork(tools.get_random_cidr())
            self._update_row(cidr, random_cidr)
            obj = self._get_one(random_cidr)
            self.assertEqual(random_cidr, obj.cidr)
        objs = self._get_all()
        self.assertEqual(len(cidrs), len(objs))
        self._delete_rows()
        objs = self._get_all()
        self.assertEqual(0, len(objs))

    def test_wrong_cidr(self):
        wrong_cidrs = ['10.500.5.0/24', '10.0.0.1/40', '10.0.0.10.0/24', 'cidr', '', '2001:db8:5000::/64', '2001:db8::/130']
        for cidr in wrong_cidrs:
            self.assertRaises(exception.DBError, self._add_row, id=uuidutils.generate_uuid(), cidr=cidr)