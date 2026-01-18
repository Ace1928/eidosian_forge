from oslo_db.sqlalchemy import test_fixtures
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestMitaka01PostgresSQL(TestMitaka01Mixin, test_fixtures.OpportunisticDBTestMixin, test_utils.BaseTestCase):
    FIXTURE = test_fixtures.PostgresqlOpportunisticFixture