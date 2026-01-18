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
def _init_impl(self):
    old_debug = _client_init_debug.get(False)

    def _capture_envelope(envelope):
        if self.transport is not None:
            self.transport.capture_envelope(envelope)
    try:
        _client_init_debug.set(self.options['debug'])
        self.transport = make_transport(self.options)
        self.monitor = None
        if self.transport:
            if self.options['enable_backpressure_handling']:
                self.monitor = Monitor(self.transport)
        self.session_flusher = SessionFlusher(capture_func=_capture_envelope)
        self.metrics_aggregator = None
        experiments = self.options.get('_experiments', {})
        if experiments.get('enable_metrics', True):
            metrics_supported = not is_gevent() or PY37
            if metrics_supported:
                from sentry_sdk.metrics import MetricsAggregator
                self.metrics_aggregator = MetricsAggregator(capture_func=_capture_envelope, enable_code_locations=bool(experiments.get('metric_code_locations', True)))
            else:
                logger.info('Metrics not supported on Python 3.6 and lower with gevent.')
        max_request_body_size = ('always', 'never', 'small', 'medium')
        if self.options['max_request_body_size'] not in max_request_body_size:
            raise ValueError('Invalid value for max_request_body_size. Must be one of {}'.format(max_request_body_size))
        if self.options['_experiments'].get('otel_powered_performance', False):
            logger.debug('[OTel] Enabling experimental OTel-powered performance monitoring.')
            self.options['instrumenter'] = INSTRUMENTER.OTEL
            _DEFAULT_INTEGRATIONS.append('sentry_sdk.integrations.opentelemetry.integration.OpenTelemetryIntegration')
        self.integrations = setup_integrations(self.options['integrations'], with_defaults=self.options['default_integrations'], with_auto_enabling_integrations=self.options['auto_enabling_integrations'])
        self.spotlight = None
        if self.options.get('spotlight'):
            self.spotlight = setup_spotlight(self.options)
        sdk_name = get_sdk_name(list(self.integrations.keys()))
        SDK_INFO['name'] = sdk_name
        logger.debug("Setting SDK name to '%s'", sdk_name)
        if has_profiling_enabled(self.options):
            try:
                setup_profiler(self.options)
            except Exception as e:
                logger.debug('Can not set up profiler. (%s)', e)
    finally:
        _client_init_debug.set(old_debug)
    self._setup_instrumentation(self.options.get('functions_to_trace', []))
    if self.monitor or self.metrics_aggregator or has_profiling_enabled(self.options) or isinstance(self.transport, HttpTransport):
        check_uwsgi_thread_support()