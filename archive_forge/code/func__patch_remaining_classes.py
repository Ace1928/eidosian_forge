import sys
from importlib import import_module
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.opentelemetry.span_processor import SentrySpanProcessor
from sentry_sdk.integrations.opentelemetry.propagator import SentryPropagator
from sentry_sdk.utils import logger, _get_installed_modules
from sentry_sdk._types import TYPE_CHECKING
def _patch_remaining_classes(original_classes):
    """
    Best-effort attempt to patch any uninstrumented classes in sys.modules.

    This enables us to not care about the order of imports and sentry_sdk.init()
    in user code. If e.g. the Flask class had been imported before sentry_sdk
    was init()ed (and therefore before the OTel instrumentation ran), it would
    not be instrumented. This function goes over remaining uninstrumented
    occurrences of the class in sys.modules and replaces them with the
    instrumented class.

    Since this is looking for exact matches, it will not work in some scenarios
    (e.g. if someone is not using the specific class explicitly, but rather
    inheriting from it). In those cases it's still necessary to sentry_sdk.init()
    before importing anything that's supposed to be instrumented.
    """
    instrumented_classes = {}
    for package in list(original_classes.keys()):
        original_path = CLASSES_TO_INSTRUMENT[package]
        try:
            cls = _import_by_path(original_path)
        except (AttributeError, ImportError):
            logger.debug('[OTel] Failed to check if class has been instrumented: %s', original_path)
            del original_classes[package]
            continue
        if not cls.__module__.startswith('opentelemetry.'):
            del original_classes[package]
            continue
        instrumented_classes[package] = cls
    if not instrumented_classes:
        return
    for module_name, module in sys.modules.copy().items():
        if module_name.startswith('sentry_sdk') or module_name in sys.builtin_module_names:
            continue
        for package, original_cls in original_classes.items():
            for var_name, var in vars(module).copy().items():
                if var == original_cls:
                    logger.debug('[OTel] Additionally patching %s from %s', original_cls, module_name)
                    setattr(module, var_name, instrumented_classes[package])