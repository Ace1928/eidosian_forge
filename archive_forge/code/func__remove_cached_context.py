import fixtures
from oslo_context import context
def _remove_cached_context(self) -> None:
    """Remove the thread-local context stored in the module."""
    try:
        del context._request_store.context
    except AttributeError:
        pass