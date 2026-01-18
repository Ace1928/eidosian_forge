import fixtures
from oslo_context import context
class ClearRequestContext(fixtures.Fixture):
    """Clears any cached RequestContext

    This resets RequestContext at the beginning and end of tests that
    use this fixture to ensure that we have a clean slate for running
    tests, and that we leave a clean slate for other tests that might
    run later in the same process.
    """

    def setUp(self) -> None:
        super(ClearRequestContext, self).setUp()
        self._remove_cached_context()
        self.addCleanup(self._remove_cached_context)

    def _remove_cached_context(self) -> None:
        """Remove the thread-local context stored in the module."""
        try:
            del context._request_store.context
        except AttributeError:
            pass