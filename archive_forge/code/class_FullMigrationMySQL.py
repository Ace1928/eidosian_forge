import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures as db_fixtures
from oslo_log import fixture as log_fixture
from oslo_log import log
import sqlalchemy.exc
from keystone.cmd import cli
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
class FullMigrationMySQL(MigrateBase, unit.TestCase):
    FIXTURE = db_fixtures.MySQLOpportunisticFixture