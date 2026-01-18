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
def add_attachment(self, bytes=None, filename=None, path=None, content_type=None, add_to_transactions=False):
    """Adds an attachment to future events sent."""
    self._attachments.append(Attachment(bytes=bytes, path=path, filename=filename, content_type=content_type, add_to_transactions=add_to_transactions))