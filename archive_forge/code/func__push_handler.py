import contextlib
import logging
import threading
from threading import local as thread_local
from threading import Thread
import traceback
from types import MethodType
import weakref
import sys
from .constants import ComparisonMode, TraitKind
from .trait_base import Uninitialized
from .trait_errors import TraitNotificationError
def _push_handler(self, handler=None, reraise_exceptions=False, main=False, locked=False):
    """ Pushes a new traits notification exception handler onto the stack,
            making it the new exception handler. Returns a
            NotificationExceptionHandlerState object describing the previous
            exception handler.

            Parameters
            ----------
            handler : handler
                The new exception handler, which should be a callable or
                None. If None (the default), then the default traits
                notification exception handler is used. If *handler* is not
                None, then it must be a callable which can accept four
                arguments: object, trait_name, old_value, new_value.
            reraise_exceptions : bool
                Indicates whether exceptions should be reraised after the
                exception handler has executed. If True, exceptions will be
                re-raised after the specified handler has been executed.
                The default value is False.
            main : bool
                Indicates whether the caller represents the main application
                thread. If True, then the caller's exception handler is
                made the default handler for any other threads that are
                created. Note that a thread can explicitly set its own
                exception handler if desired. The *main* flag is provided to
                make it easier to set a global application policy without
                having to explicitly set it for each thread. The default
                value is False.
            locked : bool
                Indicates whether further changes to the Traits notification
                exception handler state should be allowed. If True, then
                any subsequent calls to _push_handler() or _pop_handler() for
                that thread will raise a TraitNotificationError. The default
                value is False.
        """
    handlers = self._get_handlers()
    self._check_lock(handlers)
    if handler is None:
        handler = self._log_exception
    handlers.append(NotificationExceptionHandlerState(handler, reraise_exceptions, locked))
    if main:
        self.main_thread = handlers
    return handlers[-2]