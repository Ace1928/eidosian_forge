from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _notify_single_step(self, event):
    """
        Notify breakpoints of a single step exception event.

        @type  event: L{ExceptionEvent}
        @param event: Single step exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
    pid = event.get_pid()
    tid = event.get_tid()
    aThread = event.get_thread()
    aProcess = event.get_process()
    bCallHandler = True
    bIsOurs = False
    old_continueStatus = event.continueStatus
    try:
        if self.in_hostile_mode():
            event.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
        if self.system.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
            return bCallHandler
        bFakeSingleStep = False
        bLastIsPushFlags = False
        bNextIsPopFlags = False
        if self.in_hostile_mode():
            pc = aThread.get_pc()
            c = aProcess.read_char(pc - 1)
            if c == 241:
                bFakeSingleStep = True
            elif c == 156:
                bLastIsPushFlags = True
            c = aProcess.peek_char(pc)
            if c == 102:
                c = aProcess.peek_char(pc + 1)
            if c == 157:
                if bLastIsPushFlags:
                    bLastIsPushFlags = False
                else:
                    bNextIsPopFlags = True
        if self.is_tracing(tid):
            bIsOurs = True
            if not bFakeSingleStep:
                event.continueStatus = win32.DBG_CONTINUE
            aThread.set_tf()
            if bLastIsPushFlags or bNextIsPopFlags:
                sp = aThread.get_sp()
                flags = aProcess.read_dword(sp)
                if bLastIsPushFlags:
                    flags &= ~Thread.Flags.Trap
                else:
                    flags |= Thread.Flags.Trap
                aProcess.write_dword(sp, flags)
        running = self.__get_running_bp_set(tid)
        if running:
            bIsOurs = True
            if not bFakeSingleStep:
                event.continueStatus = win32.DBG_CONTINUE
            bCallHandler = False
            while running:
                try:
                    running.pop().hit(event)
                except Exception:
                    e = sys.exc_info()[1]
                    warnings.warn(str(e), BreakpointWarning)
        if tid in self.__hardwareBP:
            ctx = aThread.get_context(win32.CONTEXT_DEBUG_REGISTERS)
            Dr6 = ctx['Dr6']
            ctx['Dr6'] = Dr6 & DebugRegister.clearHitMask
            aThread.set_context(ctx)
            bFoundBreakpoint = False
            bCondition = False
            hwbpList = [bp for bp in self.__hardwareBP[tid]]
            for bp in hwbpList:
                if not bp in self.__hardwareBP[tid]:
                    continue
                slot = bp.get_slot()
                if slot is not None and Dr6 & DebugRegister.hitMask[slot]:
                    if not bFoundBreakpoint:
                        if not bFakeSingleStep:
                            event.continueStatus = win32.DBG_CONTINUE
                    bFoundBreakpoint = True
                    bIsOurs = True
                    bp.hit(event)
                    if bp.is_running():
                        self.__add_running_bp(tid, bp)
                    bThisCondition = bp.eval_condition(event)
                    if bThisCondition and bp.is_automatic():
                        bp.run_action(event)
                        bThisCondition = False
                    bCondition = bCondition or bThisCondition
            if bFoundBreakpoint:
                bCallHandler = bCondition
        if self.is_tracing(tid):
            bCallHandler = True
        if not bIsOurs and (not self.in_hostile_mode()):
            aThread.clear_tf()
    except:
        event.continueStatus = old_continueStatus
        raise
    return bCallHandler