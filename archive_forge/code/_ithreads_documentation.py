from typing import Callable
from zope.interface import Interface

        Free any resources associated with this L{IWorker} and cause it to
        reject all future work.

        @raise AlreadyQuit: if this method has already been called.
        