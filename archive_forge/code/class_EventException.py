import inspect
from functools import partial
from weakref import WeakMethod
class EventException(Exception):
    """An exception raised when an event handler could not be attached.
    """
    pass