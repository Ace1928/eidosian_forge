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
class NewTraitChangeNotifyWrapper(TraitChangeNotifyWrapper):
    """ Dynamic change notify wrapper, dispatching on a new thread.

    This class is in charge to dispatch trait change events to dynamic
    listener, typically created using the `on_trait_change` method and the
    `dispatch` parameter set to 'new'.
    """

    def dispatch(self, handler, *args):
        Thread(target=handler, args=args).start()