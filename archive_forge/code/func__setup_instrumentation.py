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
def _setup_instrumentation(self, functions_to_trace):
    """
        Instruments the functions given in the list `functions_to_trace` with the `@sentry_sdk.tracing.trace` decorator.
        """
    for function in functions_to_trace:
        class_name = None
        function_qualname = function['qualified_name']
        module_name, function_name = function_qualname.rsplit('.', 1)
        try:
            module_obj = import_module(module_name)
            function_obj = getattr(module_obj, function_name)
            setattr(module_obj, function_name, trace(function_obj))
            logger.debug('Enabled tracing for %s', function_qualname)
        except module_not_found_error:
            try:
                module_name, class_name = module_name.rsplit('.', 1)
                module_obj = import_module(module_name)
                class_obj = getattr(module_obj, class_name)
                function_obj = getattr(class_obj, function_name)
                function_type = type(class_obj.__dict__[function_name])
                traced_function = trace(function_obj)
                if function_type in (staticmethod, classmethod):
                    traced_function = staticmethod(traced_function)
                setattr(class_obj, function_name, traced_function)
                setattr(module_obj, class_name, class_obj)
                logger.debug('Enabled tracing for %s', function_qualname)
            except Exception as e:
                logger.warning("Can not enable tracing for '%s'. (%s) Please check your `functions_to_trace` parameter.", function_qualname, e)
        except Exception as e:
            logger.warning("Can not enable tracing for '%s'. (%s) Please check your `functions_to_trace` parameter.", function_qualname, e)