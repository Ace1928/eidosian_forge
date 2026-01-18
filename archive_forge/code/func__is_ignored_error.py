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
def _is_ignored_error(self, event, hint):
    exc_info = hint.get('exc_info')
    if exc_info is None:
        return False
    error = exc_info[0]
    error_type_name = get_type_name(exc_info[0])
    error_full_name = '%s.%s' % (exc_info[0].__module__, error_type_name)
    for ignored_error in self.options['ignore_errors']:
        if isinstance(ignored_error, string_types):
            if ignored_error == error_full_name or ignored_error == error_type_name:
                return True
        elif issubclass(error, ignored_error):
            return True
    return False