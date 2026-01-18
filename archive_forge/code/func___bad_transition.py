from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __bad_transition(self, state):
    """
        Raises an C{AssertionError} exception for an invalid state transition.

        @see: L{stateNames}

        @type  state: int
        @param state: Intended breakpoint state.

        @raise Exception: Always.
        """
    statemsg = ''
    oldState = self.stateNames[self.get_state()]
    newState = self.stateNames[state]
    msg = 'Invalid state transition (%s -> %s) for breakpoint at address %s'
    msg = msg % (oldState, newState, HexDump.address(self.get_address()))
    raise AssertionError(msg)