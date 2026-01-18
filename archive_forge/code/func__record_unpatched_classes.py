import sys
from importlib import import_module
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.opentelemetry.span_processor import SentrySpanProcessor
from sentry_sdk.integrations.opentelemetry.propagator import SentryPropagator
from sentry_sdk.utils import logger, _get_installed_modules
from sentry_sdk._types import TYPE_CHECKING
def _record_unpatched_classes():
    """
    Keep references to classes that are about to be instrumented.

    Used to search for unpatched classes after the instrumentation has run so
    that they can be patched manually.
    """
    installed_packages = _get_installed_modules()
    original_classes = {}
    for package, orig_path in CLASSES_TO_INSTRUMENT.items():
        if package in installed_packages:
            try:
                original_cls = _import_by_path(orig_path)
            except (AttributeError, ImportError):
                logger.debug('[OTel] Failed to import %s', orig_path)
                continue
            original_classes[package] = original_cls
    return original_classes