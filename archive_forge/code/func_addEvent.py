from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def addEvent(event: object, fd: 'FileDescriptor', action: str) -> None:
    """
        Add a new win32 event to the event loop.

        @param event: a Win32 event object created using win32event.CreateEvent()
        @param fd: an instance of L{twisted.internet.abstract.FileDescriptor}
        @param action: a string that is a method name of the fd instance.
                       This method is called in response to the event.
        """