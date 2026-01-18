from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IProcessProtocol(Interface):
    """
    Interface for process-related event handlers.
    """

    def makeConnection(process: 'IProcessTransport') -> None:
        """
        Called when the process has been created.

        @param process: An object representing the process which has been
            created and associated with this protocol.
        """

    def childDataReceived(childFD: int, data: bytes) -> None:
        """
        Called when data arrives from the child process.

        @param childFD: The file descriptor from which the data was
            received.
        @param data: The data read from the child's file descriptor.
        """

    def childConnectionLost(childFD: int) -> None:
        """
        Called when a file descriptor associated with the child process is
        closed.

        @param childFD: The file descriptor which was closed.
        """

    def processExited(reason: Failure) -> None:
        """
        Called when the child process exits.

        @param reason: A failure giving the reason the child process
            terminated.  The type of exception for this failure is either
            L{twisted.internet.error.ProcessDone} or
            L{twisted.internet.error.ProcessTerminated}.

        @since: 8.2
        """

    def processEnded(reason: Failure) -> None:
        """
        Called when the child process exits and all file descriptors associated
        with it have been closed.

        @param reason: A failure giving the reason the child process
            terminated.  The type of exception for this failure is either
            L{twisted.internet.error.ProcessDone} or
            L{twisted.internet.error.ProcessTerminated}.
        """