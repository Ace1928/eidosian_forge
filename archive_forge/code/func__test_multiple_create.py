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