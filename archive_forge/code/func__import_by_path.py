import sys
from importlib import import_module
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.opentelemetry.span_processor import SentrySpanProcessor
from sentry_sdk.integrations.opentelemetry.propagator import SentryPropagator
from sentry_sdk.utils import logger, _get_installed_modules
from sentry_sdk._types import TYPE_CHECKING
def _import_by_path(path):
    parts = path.rsplit('.', maxsplit=1)
    return getattr(import_module(parts[0]), parts[-1])