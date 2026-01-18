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
def _validate_ip_address(self, expected=None):
    objs = self._get_all()
    self.assertEqual(len(expected) if expected else 0, len(objs))
    if expected:
        for obj in objs:
            name = obj.id
            self.assertEqual(expected[name], obj.ip)