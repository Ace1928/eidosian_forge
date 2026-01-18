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
class IPAddressTestCase(SqlAlchemyTypesBaseTestCase):

    def _get_test_table(self, meta):
        return sa.Table('fakeipaddressmodels', meta, sa.Column('id', sa.String(36), primary_key=True, nullable=False), sa.Column('ip', sqlalchemytypes.IPAddress))

    def _validate_ip_address(self, expected=None):
        objs = self._get_all()
        self.assertEqual(len(expected) if expected else 0, len(objs))
        if expected:
            for obj in objs:
                name = obj.id
                self.assertEqual(expected[name], obj.ip)

    def _test_crud(self, ip_addresses):
        ip = netaddr.IPAddress(ip_addresses[0])
        self._add_row(id='fake_id', ip=ip)
        self._validate_ip_address(expected={'fake_id': ip})
        ip2 = netaddr.IPAddress(ip_addresses[1])
        self._update_row(ip=ip2)
        self._validate_ip_address(expected={'fake_id': ip2})
        self._delete_rows()
        self._validate_ip_address(expected=None)

    def test_crud(self):
        ip_addresses = ['10.0.0.1', '10.0.0.2']
        self._test_crud(ip_addresses)
        ip_addresses = ['2210::ffff:ffff:ffff:ffff', '2120::ffff:ffff:ffff:ffff']
        self._test_crud(ip_addresses)

    def test_wrong_type(self):
        self.assertRaises(exception.DBError, self._add_row, id='fake_id', ip='')
        self.assertRaises(exception.DBError, self._add_row, id='fake_id', ip='10.0.0.5')

    def _test_multiple_create(self, entries):
        reference = {}
        for entry in entries:
            ip = netaddr.IPAddress(entry['ip'])
            name = entry['name']
            self._add_row(id=name, ip=ip)
            reference[name] = ip
        self._validate_ip_address(expected=reference)
        self._delete_rows()
        self._validate_ip_address(expected=None)

    def test_multiple_create(self):
        ip_addresses = [{'name': 'fake_id1', 'ip': '10.0.0.5'}, {'name': 'fake_id2', 'ip': '10.0.0.1'}, {'name': 'fake_id3', 'ip': '2210::ffff:ffff:ffff:ffff'}, {'name': 'fake_id4', 'ip': '2120::ffff:ffff:ffff:ffff'}]
        self._test_multiple_create(ip_addresses)