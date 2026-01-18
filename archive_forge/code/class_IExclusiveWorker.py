from typing import Callable
from zope.interface import Interface
class IExclusiveWorker(IWorker):
    """
    Like L{IWorker}, but with the additional guarantee that the callables
    passed to C{do} will not be called exclusively with each other.
    """