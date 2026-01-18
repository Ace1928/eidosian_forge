from __future__ import absolute_import
import copy
from sentry_sdk import Hub
from sentry_sdk.consts import SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _operation_key(self, event):
    return event.request_id