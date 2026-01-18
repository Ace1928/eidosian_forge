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
def _merge_scopes(base, scope_change, scope_kwargs):
    if scope_change and scope_kwargs:
        raise TypeError('cannot provide scope and kwargs')
    if scope_change is not None:
        final_scope = copy(base)
        if callable(scope_change):
            scope_change(final_scope)
        else:
            final_scope.update_from_scope(scope_change)
    elif scope_kwargs:
        final_scope = copy(base)
        final_scope.update_from_kwargs(**scope_kwargs)
    else:
        final_scope = base
    return final_scope