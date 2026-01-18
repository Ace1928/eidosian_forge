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
def add_event_processor(self, func):
    """Register a scope local event processor on the scope.

        :param func: This function behaves like `before_send.`
        """
    if len(self._event_processors) > 20:
        logger.warning('Too many event processors on scope! Clearing list to free up some memory: %r', self._event_processors)
        del self._event_processors[:]
    self._event_processors.append(func)