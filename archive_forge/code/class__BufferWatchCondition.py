from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class _BufferWatchCondition(object):
    """
    Used by L{Debug.watch_buffer}.

    This class acts as a condition callback for page breakpoints.
    It emulates page breakpoints that can overlap and/or take up less
    than a page's size.
    """

    def __init__(self):
        self.__ranges = list()

    def add(self, bw):
        """
        Adds a buffer watch identifier.

        @type  bw: L{BufferWatch}
        @param bw:
            Buffer watch identifier.
        """
        self.__ranges.append(bw)

    def remove(self, bw):
        """
        Removes a buffer watch identifier.

        @type  bw: L{BufferWatch}
        @param bw:
            Buffer watch identifier.

        @raise KeyError: The buffer watch identifier was already removed.
        """
        try:
            self.__ranges.remove(bw)
        except KeyError:
            if not bw.oneshot:
                raise

    def remove_last_match(self, address, size):
        """
        Removes the last buffer from the watch object
        to match the given address and size.

        @type  address: int
        @param address: Memory address of buffer to stop watching.

        @type  size: int
        @param size: Size in bytes of buffer to stop watching.

        @rtype:  int
        @return: Number of matching elements found. Only the last one to be
            added is actually deleted upon calling this method.

            This counter allows you to know if there are more matching elements
            and how many.
        """
        count = 0
        start = address
        end = address + size - 1
        matched = None
        for item in self.__ranges:
            if item.match(start) and item.match(end):
                matched = item
                count += 1
        self.__ranges.remove(matched)
        return count

    def count(self):
        """
        @rtype:  int
        @return: Number of buffers being watched.
        """
        return len(self.__ranges)

    def __call__(self, event):
        """
        Breakpoint condition callback.

        This method will also call the action callbacks for each
        buffer being watched.

        @type  event: L{ExceptionEvent}
        @param event: Guard page exception event.

        @rtype:  bool
        @return: C{True} if the address being accessed belongs
            to at least one of the buffers that was being watched
            and had no action callback.
        """
        address = event.get_exception_information(1)
        bCondition = False
        for bw in self.__ranges:
            bMatched = bw.match(address)
            try:
                action = bw.action
                if bMatched and action is not None:
                    try:
                        action(event)
                    except Exception:
                        e = sys.exc_info()[1]
                        msg = 'Breakpoint action callback %r raised an exception: %s'
                        msg = msg % (action, traceback.format_exc(e))
                        warnings.warn(msg, BreakpointCallbackWarning)
                else:
                    bCondition = bCondition or bMatched
            finally:
                if bMatched and bw.oneshot:
                    event.debug.dont_watch_buffer(bw)
        return bCondition