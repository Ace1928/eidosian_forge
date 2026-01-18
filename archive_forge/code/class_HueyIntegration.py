from __future__ import absolute_import
import sys
from datetime import datetime
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.api import continue_trace, get_baggage, get_traceparent
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
class HueyIntegration(Integration):
    identifier = 'huey'

    @staticmethod
    def setup_once():
        patch_enqueue()
        patch_execute()