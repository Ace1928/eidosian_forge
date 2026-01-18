from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
class ClearRequestContextTest(test_base.BaseTestCase):

    def test_store_current(self):
        ctx = context.RequestContext()
        self.assertIs(context.get_current(), ctx)
        fixture.ClearRequestContext()._remove_cached_context()
        self.assertIsNone(context.get_current())

    def test_store_current_resets_correctly(self):
        ctx = context.RequestContext()
        self.useFixture(fixture.ClearRequestContext())
        self.assertIsNone(context.get_current())
        ctx = context.RequestContext()
        self.assertIs(context.get_current(), ctx)
        fixture.ClearRequestContext()._remove_cached_context()
        self.assertIsNone(context.get_current())