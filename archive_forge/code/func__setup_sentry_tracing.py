import sys
from importlib import import_module
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.opentelemetry.span_processor import SentrySpanProcessor
from sentry_sdk.integrations.opentelemetry.propagator import SentryPropagator
from sentry_sdk.utils import logger, _get_installed_modules
from sentry_sdk._types import TYPE_CHECKING
def _setup_sentry_tracing():
    provider = TracerProvider()
    provider.add_span_processor(SentrySpanProcessor())
    trace.set_tracer_provider(provider)
    set_global_textmap(SentryPropagator())