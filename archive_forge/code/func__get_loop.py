import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
def _get_loop(fut):
    try:
        get_loop = fut.get_loop
    except AttributeError:
        pass
    else:
        return get_loop()
    return fut._loop