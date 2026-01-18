import codecs
import contextlib
import locale
import logging
import math
import os
from functools import partial
from typing import TextIO, Union
import dill
class TraceAdapter(logging.LoggerAdapter):
    """
    Tracks object tree depth and calculates pickled object size.

    A single instance of this wraps the module's logger, as the logging API
    doesn't allow setting it directly with a custom Logger subclass.  The added
    'trace()' method receives a pickle instance as the first argument and
    creates extra values to be added in the LogRecord from it, then calls
    'info()'.

    Usage of logger with 'trace()' method:

    >>> from dill.logger import adapter as logger  #NOTE: not dill.logger.logger
    >>> ...
    >>> def save_atype(pickler, obj):
    >>>     logger.trace(pickler, "Message with %s and %r etc. placeholders", 'text', obj)
    >>>     ...
    """

    def __init__(self, logger):
        self.logger = logger

    def addHandler(self, handler):
        formatter = TraceFormatter('%(prefix)s%(message)s%(suffix)s', handler=handler)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def removeHandler(self, handler):
        self.logger.removeHandler(handler)

    def process(self, msg, kwargs):
        return (msg, kwargs)

    def trace_setup(self, pickler):
        if not dill._dill.is_dill(pickler, child=False):
            return
        if self.isEnabledFor(logging.INFO):
            pickler._trace_depth = 1
            pickler._size_stack = []
        else:
            pickler._trace_depth = None

    def trace(self, pickler, msg, *args, **kwargs):
        if not hasattr(pickler, '_trace_depth'):
            logger.info(msg, *args, **kwargs)
            return
        if pickler._trace_depth is None:
            return
        extra = kwargs.get('extra', {})
        pushed_obj = msg.startswith('#')
        size = None
        try:
            size = pickler._file.tell()
            frame = pickler.framer.current_frame
            try:
                size += frame.tell()
            except AttributeError:
                size += len(frame)
        except (AttributeError, TypeError):
            pass
        if size is not None:
            if not pushed_obj:
                pickler._size_stack.append(size)
            else:
                size -= pickler._size_stack.pop()
                extra['size'] = size
        if pushed_obj:
            pickler._trace_depth -= 1
        extra['depth'] = pickler._trace_depth
        kwargs['extra'] = extra
        self.info(msg, *args, **kwargs)
        if not pushed_obj:
            pickler._trace_depth += 1