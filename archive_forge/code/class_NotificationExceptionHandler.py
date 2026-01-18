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
class NotificationExceptionHandler(object):

    def __init__(self):
        self.traits_logger = None
        self.main_thread = None
        self.thread_local = thread_local()

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

    def _pop_handler(self):
        """ Pops the traits notification exception handler stack, restoring
            the exception handler in effect prior to the most recent
            _push_handler() call. If the stack is empty or locked, a
            TraitNotificationError exception is raised.

            Note that each thread has its own independent stack. See the
            description of the _push_handler() method for more information on
            this.
        """
        handlers = self._get_handlers()
        self._check_lock(handlers)
        if len(handlers) > 1:
            handlers.pop()
        else:
            raise TraitNotificationError('Attempted to pop an empty traits notification exception handler stack.')

    def _handle_exception(self, object, trait_name, old, new):
        """ Handles a traits notification exception using the handler defined
            by the topmost stack entry for the corresponding thread.
        """
        excp_class, excp = sys.exc_info()[:2]
        handler_info = self._get_handlers()[-1]
        handler_info.handler(object, trait_name, old, new)
        if handler_info.reraise_exceptions or isinstance(excp, TraitNotificationError):
            raise excp

    def _get_handlers(self):
        """ Returns the handler stack associated with the currently executing
            thread.
        """
        thread_local = self.thread_local
        if isinstance(thread_local, dict):
            id = threading.current_thread().ident
            handlers = thread_local.get(id)
        else:
            handlers = getattr(thread_local, 'handlers', None)
        if handlers is None:
            if self.main_thread is not None:
                handler = self.main_thread[-1]
            else:
                handler = NotificationExceptionHandlerState(self._log_exception, False, False)
            handlers = [handler]
            if isinstance(thread_local, dict):
                thread_local[id] = handlers
            else:
                thread_local.handlers = handlers
        return handlers

    def _check_lock(self, handlers):
        """ Raises an exception if the specified handler stack is locked.
        """
        if handlers[-1].locked:
            raise TraitNotificationError('The traits notification exception handler is locked. No changes are allowed.')

    def _log_exception(self, object, trait_name, old, new):
        """ Logs any exceptions generated in a trait notification handler.

        This method defines the default notification exception handling
        behavior of traits. However, it can be completely overridden by pushing
        a new handler using the '_push_handler' method.
        """
        excp_class, excp = sys.exc_info()[:2]
        if excp_class is RuntimeError and len(excp.args) > 0 and (excp.args[0] == 'maximum recursion depth exceeded'):
            sys.__stderr__.write('Exception occurred in traits notification handler for object: %s, trait: %s, old value: %s, new value: %s.\n%s\n' % (object, trait_name, old, new, ''.join(traceback.format_exception(*sys.exc_info()))))
        logger = self.traits_logger
        if logger is None:
            self.traits_logger = logger = logging.getLogger('traits')
        try:
            logger.exception('Exception occurred in traits notification handler for object: %s, trait: %s, old value: %s, new value: %s' % (object, trait_name, old, new))
        except Exception:
            pass