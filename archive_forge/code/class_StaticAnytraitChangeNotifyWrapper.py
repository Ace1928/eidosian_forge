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
class StaticAnytraitChangeNotifyWrapper(AbstractStaticChangeNotifyWrapper):
    argument_transforms = {0: lambda obj, name, old, new: (), 1: lambda obj, name, old, new: (obj,), 2: lambda obj, name, old, new: (obj, name), 3: lambda obj, name, old, new: (obj, name, new), 4: lambda obj, name, old, new: (obj, name, old, new)}