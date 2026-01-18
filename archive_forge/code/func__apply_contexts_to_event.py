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
def _apply_contexts_to_event(self, event, hint, options):
    if self._contexts:
        event.setdefault('contexts', {}).update(self._contexts)
    contexts = event.setdefault('contexts', {})
    if contexts.get('trace') is None:
        if has_tracing_enabled(options) and self._span is not None:
            contexts['trace'] = self._span.get_trace_context()
        else:
            contexts['trace'] = self.get_trace_context()
    try:
        replay_id = contexts['trace']['dynamic_sampling_context']['replay_id']
    except (KeyError, TypeError):
        replay_id = None
    if replay_id is not None:
        contexts['replay'] = {'replay_id': replay_id}