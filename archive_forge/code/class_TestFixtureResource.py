from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class TestFixtureResource(testtools.TestCase):

    def test_uses_setUp_cleanUp(self):
        fixture = LoggingFixture()
        mgr = testresources.FixtureResource(fixture)
        resource = mgr.getResource()
        self.assertEqual(fixture, resource)
        self.assertEqual(['setUp'], fixture.calls)
        mgr.finishedWith(resource)
        self.assertEqual(['setUp', 'cleanUp'], fixture.calls)

    def test_always_dirty(self):
        fixture = LoggingFixture()
        mgr = testresources.FixtureResource(fixture)
        resource = mgr.getResource()
        self.assertTrue(mgr.isDirty())
        mgr.finishedWith(resource)

    def test_reset_called(self):
        fixture = LoggingFixture()
        mgr = testresources.FixtureResource(fixture)
        resource = mgr.getResource()
        mgr.reset(resource)
        mgr.finishedWith(resource)
        self.assertEqual(['setUp', 'reset', 'cleanUp'], fixture.calls)