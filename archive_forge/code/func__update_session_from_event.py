from importlib import import_module
import os
import uuid
import random
import socket
from sentry_sdk._compat import (
from sentry_sdk.utils import (
from sentry_sdk.serializer import serialize
from sentry_sdk.tracing import trace, has_tracing_enabled
from sentry_sdk.transport import HttpTransport, make_transport
from sentry_sdk.consts import (
from sentry_sdk.integrations import _DEFAULT_INTEGRATIONS, setup_integrations
from sentry_sdk.utils import ContextVar
from sentry_sdk.sessions import SessionFlusher
from sentry_sdk.envelope import Envelope
from sentry_sdk.profiler import has_profiling_enabled, Profile, setup_profiler
from sentry_sdk.scrubber import EventScrubber
from sentry_sdk.monitor import Monitor
from sentry_sdk.spotlight import setup_spotlight
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._types import TYPE_CHECKING
def _update_session_from_event(self, session, event):
    crashed = False
    errored = False
    user_agent = None
    exceptions = (event.get('exception') or {}).get('values')
    if exceptions:
        errored = True
        for error in exceptions:
            mechanism = error.get('mechanism')
            if isinstance(mechanism, Mapping) and mechanism.get('handled') is False:
                crashed = True
                break
    user = event.get('user')
    if session.user_agent is None:
        headers = (event.get('request') or {}).get('headers')
        for k, v in iteritems(headers or {}):
            if k.lower() == 'user-agent':
                user_agent = v
                break
    session.update(status='crashed' if crashed else None, user=user, user_agent=user_agent, errors=session.errors + (errored or crashed))