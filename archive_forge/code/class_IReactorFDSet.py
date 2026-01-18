from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorFDSet(Interface):
    """
    Implement me to be able to use L{IFileDescriptor} type resources.

    This assumes that your main-loop uses UNIX-style numeric file descriptors
    (or at least similarly opaque IDs returned from a .fileno() method)
    """

    def addReader(reader: 'IReadDescriptor') -> None:
        """
        I add reader to the set of file descriptors to get read events for.

        @param reader: An L{IReadDescriptor} provider that will be checked for
                       read events until it is removed from the reactor with
                       L{removeReader}.
        """

    def addWriter(writer: 'IWriteDescriptor') -> None:
        """
        I add writer to the set of file descriptors to get write events for.

        @param writer: An L{IWriteDescriptor} provider that will be checked for
                       write events until it is removed from the reactor with
                       L{removeWriter}.
        """

    def removeReader(reader: 'IReadDescriptor') -> None:
        """
        Removes an object previously added with L{addReader}.
        """

    def removeWriter(writer: 'IWriteDescriptor') -> None:
        """
        Removes an object previously added with L{addWriter}.
        """

    def removeAll() -> List[Union['IReadDescriptor', 'IWriteDescriptor']]:
        """
        Remove all readers and writers.

        Should not remove reactor internal reactor connections (like a waker).

        @return: A list of L{IReadDescriptor} and L{IWriteDescriptor} providers
                 which were removed.
        """

    def getReaders() -> List['IReadDescriptor']:
        """
        Return the list of file descriptors currently monitored for input
        events by the reactor.

        @return: the list of file descriptors monitored for input events.
        """

    def getWriters() -> List['IWriteDescriptor']:
        """
        Return the list file descriptors currently monitored for output events
        by the reactor.

        @return: the list of file descriptors monitored for output events.
        """