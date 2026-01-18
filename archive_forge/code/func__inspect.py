from __future__ import absolute_import
import sys
import types
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub
from sentry_sdk._compat import reraise
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
def _inspect(self):
    """
        Inspect function overrides the way Beam gets argspec.
        """
    wrapped_func = WRAPPED_FUNC.format(func_name)
    if hasattr(self, wrapped_func):
        process_func = getattr(self, wrapped_func)
    else:
        process_func = getattr(self, func_name)
        setattr(self, func_name, _wrap_task_call(process_func))
        setattr(self, wrapped_func, process_func)
    try:
        from apache_beam.transforms.core import get_function_args_defaults
        return get_function_args_defaults(process_func)
    except ImportError:
        from apache_beam.typehints.decorators import getfullargspec
        return getfullargspec(process_func)