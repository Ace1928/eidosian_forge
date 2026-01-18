import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
class FixtureWithException(fixtures.Fixture):

    def setUp(self):
        super(FixtureWithException, self).setUp()
        self.addCleanup(raise_exception2)
        self.addCleanup(raise_exception1)