import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
def _call_set_state(source):
    if destination.cancelled() and dest_loop is not None and dest_loop.is_closed():
        return
    if dest_loop is None or dest_loop is source_loop:
        _set_state(destination, source)
    else:
        if dest_loop.is_closed():
            return
        dest_loop.call_soon_threadsafe(_set_state, destination, source)