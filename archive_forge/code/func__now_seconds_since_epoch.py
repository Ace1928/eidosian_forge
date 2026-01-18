from __future__ import absolute_import
import sys
import time
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk.crons import capture_checkin, MonitorStatus
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import BAGGAGE_HEADER_NAME, TRANSACTION_SOURCE_TASK
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _now_seconds_since_epoch():
    return time.time()