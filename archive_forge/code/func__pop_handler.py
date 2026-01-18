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