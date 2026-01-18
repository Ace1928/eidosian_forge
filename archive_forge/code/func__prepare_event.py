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
def _prepare_event(self, event, hint, scope):
    if event.get('timestamp') is None:
        event['timestamp'] = datetime_utcnow()
    if scope is not None:
        is_transaction = event.get('type') == 'transaction'
        event_ = scope.apply_to_event(event, hint, self.options)
        if event_ is None:
            if self.transport:
                self.transport.record_lost_event('event_processor', data_category='transaction' if is_transaction else 'error')
            return None
        event = event_
    if self.options['attach_stacktrace'] and 'exception' not in event and ('stacktrace' not in event) and ('threads' not in event):
        with capture_internal_exceptions():
            event['threads'] = {'values': [{'stacktrace': current_stacktrace(include_local_variables=self.options.get('include_local_variables', True), max_value_length=self.options.get('max_value_length', DEFAULT_MAX_VALUE_LENGTH)), 'crashed': False, 'current': True}]}
    for key in ('release', 'environment', 'server_name', 'dist'):
        if event.get(key) is None and self.options[key] is not None:
            event[key] = text_type(self.options[key]).strip()
    if event.get('sdk') is None:
        sdk_info = dict(SDK_INFO)
        sdk_info['integrations'] = sorted(self.integrations.keys())
        event['sdk'] = sdk_info
    if event.get('platform') is None:
        event['platform'] = 'python'
    event = handle_in_app(event, self.options['in_app_exclude'], self.options['in_app_include'], self.options['project_root'])
    if event is not None:
        event_scrubber = self.options['event_scrubber']
        if event_scrubber and (not self.options['send_default_pii']):
            event_scrubber.scrub_event(event)
    if event is not None:
        event = serialize(event, max_request_body_size=self.options.get('max_request_body_size'), max_value_length=self.options.get('max_value_length'))
    before_send = self.options['before_send']
    if before_send is not None and event is not None and (event.get('type') != 'transaction'):
        new_event = None
        with capture_internal_exceptions():
            new_event = before_send(event, hint or {})
        if new_event is None:
            logger.info('before send dropped event')
            if self.transport:
                self.transport.record_lost_event('before_send', data_category='error')
        event = new_event
    before_send_transaction = self.options['before_send_transaction']
    if before_send_transaction is not None and event is not None and (event.get('type') == 'transaction'):
        new_event = None
        with capture_internal_exceptions():
            new_event = before_send_transaction(event, hint or {})
        if new_event is None:
            logger.info('before send transaction dropped event')
            if self.transport:
                self.transport.record_lost_event('before_send', data_category='transaction')
        event = new_event
    return event