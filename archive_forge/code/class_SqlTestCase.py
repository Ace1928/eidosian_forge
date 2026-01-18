from neutron_lib import fixture
from neutron_lib.tests import _base as base
class SqlTestCase(base.BaseTestCase):

    def setUp(self):
        super(SqlTestCase, self).setUp()
        self.useFixture(fixture.SqlFixture())