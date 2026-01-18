from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
class EventFactory(StaticClass):
    """
    Factory of L{Event} objects.

    @type baseEvent: L{Event}
    @cvar baseEvent:
        Base class for Event objects.
        It's used for unknown event codes.

    @type eventClasses: dict( int S{->} L{Event} )
    @cvar eventClasses:
        Dictionary that maps event codes to L{Event} subclasses.
    """
    baseEvent = Event
    eventClasses = {win32.EXCEPTION_DEBUG_EVENT: ExceptionEvent, win32.CREATE_THREAD_DEBUG_EVENT: CreateThreadEvent, win32.CREATE_PROCESS_DEBUG_EVENT: CreateProcessEvent, win32.EXIT_THREAD_DEBUG_EVENT: ExitThreadEvent, win32.EXIT_PROCESS_DEBUG_EVENT: ExitProcessEvent, win32.LOAD_DLL_DEBUG_EVENT: LoadDLLEvent, win32.UNLOAD_DLL_DEBUG_EVENT: UnloadDLLEvent, win32.OUTPUT_DEBUG_STRING_EVENT: OutputDebugStringEvent, win32.RIP_EVENT: RIPEvent}

    @classmethod
    def get(cls, debug, raw):
        """
        @type  debug: L{Debug}
        @param debug: Debug object that received the event.

        @type  raw: L{DEBUG_EVENT}
        @param raw: Raw DEBUG_EVENT structure as used by the Win32 API.

        @rtype: L{Event}
        @returns: An Event object or one of it's subclasses,
            depending on the event type.
        """
        eventClass = cls.eventClasses.get(raw.dwDebugEventCode, cls.baseEvent)
        return eventClass(debug, raw)