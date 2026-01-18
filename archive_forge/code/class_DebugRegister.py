import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
class DebugRegister(StaticClass):
    """
    Class to manipulate debug registers.
    Used by L{HardwareBreakpoint}.

    @group Trigger flags used by HardwareBreakpoint:
        BREAK_ON_EXECUTION, BREAK_ON_WRITE, BREAK_ON_ACCESS, BREAK_ON_IO_ACCESS
    @group Size flags used by HardwareBreakpoint:
        WATCH_BYTE, WATCH_WORD, WATCH_DWORD, WATCH_QWORD
    @group Bitwise masks for Dr7:
        enableMask, disableMask, triggerMask, watchMask, clearMask,
        generalDetectMask
    @group Bitwise masks for Dr6:
        hitMask, hitMaskAll, debugAccessMask, singleStepMask, taskSwitchMask,
        clearDr6Mask, clearHitMask
    @group Debug control MSR definitions:
        DebugCtlMSR, LastBranchRecord, BranchTrapFlag, PinControl,
        LastBranchToIP, LastBranchFromIP,
        LastExceptionToIP, LastExceptionFromIP

    @type BREAK_ON_EXECUTION: int
    @cvar BREAK_ON_EXECUTION: Break on execution.

    @type BREAK_ON_WRITE: int
    @cvar BREAK_ON_WRITE: Break on write.

    @type BREAK_ON_ACCESS: int
    @cvar BREAK_ON_ACCESS: Break on read or write.

    @type BREAK_ON_IO_ACCESS: int
    @cvar BREAK_ON_IO_ACCESS: Break on I/O port access.
        Not supported by any hardware.

    @type WATCH_BYTE: int
    @cvar WATCH_BYTE: Watch a byte.

    @type WATCH_WORD: int
    @cvar WATCH_WORD: Watch a word.

    @type WATCH_DWORD: int
    @cvar WATCH_DWORD: Watch a double word.

    @type WATCH_QWORD: int
    @cvar WATCH_QWORD: Watch one quad word.

    @type enableMask: 4-tuple of integers
    @cvar enableMask:
        Enable bit on C{Dr7} for each slot.
        Works as a bitwise-OR mask.

    @type disableMask: 4-tuple of integers
    @cvar disableMask:
        Mask of the enable bit on C{Dr7} for each slot.
        Works as a bitwise-AND mask.

    @type triggerMask: 4-tuple of 2-tuples of integers
    @cvar triggerMask:
        Trigger bits on C{Dr7} for each trigger flag value.
        Each 2-tuple has the bitwise-OR mask and the bitwise-AND mask.

    @type watchMask: 4-tuple of 2-tuples of integers
    @cvar watchMask:
        Watch bits on C{Dr7} for each watch flag value.
        Each 2-tuple has the bitwise-OR mask and the bitwise-AND mask.

    @type clearMask: 4-tuple of integers
    @cvar clearMask:
        Mask of all important bits on C{Dr7} for each slot.
        Works as a bitwise-AND mask.

    @type generalDetectMask: integer
    @cvar generalDetectMask:
        General detect mode bit. It enables the processor to notify the
        debugger when the debugee is trying to access one of the debug
        registers.

    @type hitMask: 4-tuple of integers
    @cvar hitMask:
        Hit bit on C{Dr6} for each slot.
        Works as a bitwise-AND mask.

    @type hitMaskAll: integer
    @cvar hitMaskAll:
        Bitmask for all hit bits in C{Dr6}. Useful to know if at least one
        hardware breakpoint was hit, or to clear the hit bits only.

    @type clearHitMask: integer
    @cvar clearHitMask:
        Bitmask to clear all the hit bits in C{Dr6}.

    @type debugAccessMask: integer
    @cvar debugAccessMask:
        The debugee tried to access a debug register. Needs bit
        L{generalDetectMask} enabled in C{Dr7}.

    @type singleStepMask: integer
    @cvar singleStepMask:
        A single step exception was raised. Needs the trap flag enabled.

    @type taskSwitchMask: integer
    @cvar taskSwitchMask:
        A task switch has occurred. Needs the TSS T-bit set to 1.

    @type clearDr6Mask: integer
    @cvar clearDr6Mask:
        Bitmask to clear all meaningful bits in C{Dr6}.
    """
    BREAK_ON_EXECUTION = 0
    BREAK_ON_WRITE = 1
    BREAK_ON_ACCESS = 3
    BREAK_ON_IO_ACCESS = 2
    WATCH_BYTE = 0
    WATCH_WORD = 1
    WATCH_DWORD = 3
    WATCH_QWORD = 2
    registerMask = _registerMask
    enableMask = (1 << 0, 1 << 2, 1 << 4, 1 << 6)
    disableMask = tuple([_registerMask ^ x for x in enableMask])
    try:
        del x
    except:
        pass
    triggerMask = (((0 << 16, 3 << 16 ^ registerMask), (1 << 16, 3 << 16 ^ registerMask), (2 << 16, 3 << 16 ^ registerMask), (3 << 16, 3 << 16 ^ registerMask)), ((0 << 20, 3 << 20 ^ registerMask), (1 << 20, 3 << 20 ^ registerMask), (2 << 20, 3 << 20 ^ registerMask), (3 << 20, 3 << 20 ^ registerMask)), ((0 << 24, 3 << 24 ^ registerMask), (1 << 24, 3 << 24 ^ registerMask), (2 << 24, 3 << 24 ^ registerMask), (3 << 24, 3 << 24 ^ registerMask)), ((0 << 28, 3 << 28 ^ registerMask), (1 << 28, 3 << 28 ^ registerMask), (2 << 28, 3 << 28 ^ registerMask), (3 << 28, 3 << 28 ^ registerMask)))
    watchMask = (((0 << 18, 3 << 18 ^ registerMask), (1 << 18, 3 << 18 ^ registerMask), (2 << 18, 3 << 18 ^ registerMask), (3 << 18, 3 << 18 ^ registerMask)), ((0 << 23, 3 << 23 ^ registerMask), (1 << 23, 3 << 23 ^ registerMask), (2 << 23, 3 << 23 ^ registerMask), (3 << 23, 3 << 23 ^ registerMask)), ((0 << 26, 3 << 26 ^ registerMask), (1 << 26, 3 << 26 ^ registerMask), (2 << 26, 3 << 26 ^ registerMask), (3 << 26, 3 << 26 ^ registerMask)), ((0 << 30, 3 << 31 ^ registerMask), (1 << 30, 3 << 31 ^ registerMask), (2 << 30, 3 << 31 ^ registerMask), (3 << 30, 3 << 31 ^ registerMask)))
    clearMask = (registerMask ^ (1 << 0) + (3 << 16) + (3 << 18), registerMask ^ (1 << 2) + (3 << 20) + (3 << 22), registerMask ^ (1 << 4) + (3 << 24) + (3 << 26), registerMask ^ (1 << 6) + (3 << 28) + (3 << 30))
    generalDetectMask = 1 << 13
    hitMask = (1 << 0, 1 << 1, 1 << 2, 1 << 3)
    hitMaskAll = hitMask[0] | hitMask[1] | hitMask[2] | hitMask[3]
    clearHitMask = registerMask ^ hitMaskAll
    debugAccessMask = 1 << 13
    singleStepMask = 1 << 14
    taskSwitchMask = 1 << 15
    clearDr6Mask = registerMask ^ (hitMaskAll | debugAccessMask | singleStepMask | taskSwitchMask)
    DebugCtlMSR = 473
    LastBranchRecord = 1 << 0
    BranchTrapFlag = 1 << 1
    PinControl = (1 << 2, 1 << 3, 1 << 4, 1 << 5)
    LastBranchToIP = 476
    LastBranchFromIP = 475
    LastExceptionToIP = 478
    LastExceptionFromIP = 477

    @classmethod
    def clear_bp(cls, ctx, register):
        """
        Clears a hardware breakpoint.

        @see: find_slot, set_bp

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @type  register: int
        @param register: Slot (debug register) for hardware breakpoint.
        """
        ctx['Dr7'] &= cls.clearMask[register]
        ctx['Dr%d' % register] = 0

    @classmethod
    def set_bp(cls, ctx, register, address, trigger, watch):
        """
        Sets a hardware breakpoint.

        @see: clear_bp, find_slot

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @type  register: int
        @param register: Slot (debug register).

        @type  address: int
        @param address: Memory address.

        @type  trigger: int
        @param trigger: Trigger flag. See L{HardwareBreakpoint.validTriggers}.

        @type  watch: int
        @param watch: Watch flag. See L{HardwareBreakpoint.validWatchSizes}.
        """
        Dr7 = ctx['Dr7']
        Dr7 |= cls.enableMask[register]
        orMask, andMask = cls.triggerMask[register][trigger]
        Dr7 &= andMask
        Dr7 |= orMask
        orMask, andMask = cls.watchMask[register][watch]
        Dr7 &= andMask
        Dr7 |= orMask
        ctx['Dr7'] = Dr7
        ctx['Dr%d' % register] = address

    @classmethod
    def find_slot(cls, ctx):
        """
        Finds an empty slot to set a hardware breakpoint.

        @see: clear_bp, set_bp

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @rtype:  int
        @return: Slot (debug register) for hardware breakpoint.
        """
        Dr7 = ctx['Dr7']
        slot = 0
        for m in cls.enableMask:
            if Dr7 & m == 0:
                return slot
            slot += 1
        return None