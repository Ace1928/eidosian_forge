from copy import copy
from collections import deque
from itertools import chain
import os
import sys
import uuid
from sentry_sdk.attachments import Attachment
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import FALSE_VALUES, INSTRUMENTER
from sentry_sdk._functools import wraps
from sentry_sdk.profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _capture_internal_exception(self, exc_info):
    """
        Capture an exception that is likely caused by a bug in the SDK
        itself.

        These exceptions do not end up in Sentry and are just logged instead.
        """
    logger.error('Internal error in sentry_sdk', exc_info=exc_info)