from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
@contextmanager
def _fix_sentry_scope(self):
    hub = sentry_sdk.Hub.current
    with sentry_sdk.Hub(hub) as hub:
        with hub.configure_scope() as scope:
            scope.clear_breadcrumbs()
            scope.add_event_processor(self._make_sentry_event_processor())
            yield