from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy.test_base import backend_specific  # noqa
from oslo_db.sqlalchemy import test_fixtures as db_fixtures
from oslo_db.tests import base as test_base
class _MySQLOpportunisticTestCase(_DbTestCase):
    FIXTURE = db_fixtures.MySQLOpportunisticFixture