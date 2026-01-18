import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
def _convert_future_exc(exc):
    exc_class = type(exc)
    if exc_class is concurrent.futures.CancelledError:
        return exceptions.CancelledError(*exc.args)
    elif exc_class is concurrent.futures.TimeoutError:
        return exceptions.TimeoutError(*exc.args)
    elif exc_class is concurrent.futures.InvalidStateError:
        return exceptions.InvalidStateError(*exc.args)
    else:
        return exc