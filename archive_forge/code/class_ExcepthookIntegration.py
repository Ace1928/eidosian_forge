import sys
from sentry_sdk.hub import Hub
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
class ExcepthookIntegration(Integration):
    identifier = 'excepthook'
    always_run = False

    def __init__(self, always_run=False):
        if not isinstance(always_run, bool):
            raise ValueError('Invalid value for always_run: %s (must be type boolean)' % (always_run,))
        self.always_run = always_run

    @staticmethod
    def setup_once():
        sys.excepthook = _make_excepthook(sys.excepthook)