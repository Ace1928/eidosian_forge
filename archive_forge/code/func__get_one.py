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
def _get_one(self, value):
    row_select = self.test_table.select().where(self.test_table.c.mac == value)
    with self.engine.connect() as conn, conn.begin():
        return conn.execute(row_select).first()