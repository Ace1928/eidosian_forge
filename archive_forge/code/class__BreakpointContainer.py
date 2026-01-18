from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class _BreakpointContainer(object):
    """
    Encapsulates the capability to contain Breakpoint objects.

    @group Breakpoints:
        break_at, watch_variable, watch_buffer, hook_function,
        dont_break_at, dont_watch_variable, dont_watch_buffer,
        dont_hook_function, unhook_function,
        break_on_error, dont_break_on_error

    @group Stalking:
        stalk_at, stalk_variable, stalk_buffer, stalk_function,
        dont_stalk_at, dont_stalk_variable, dont_stalk_buffer,
        dont_stalk_function

    @group Tracing:
        is_tracing, get_traced_tids,
        start_tracing, stop_tracing,
        start_tracing_process, stop_tracing_process,
        start_tracing_all, stop_tracing_all

    @group Symbols:
        resolve_label, resolve_exported_function

    @group Advanced breakpoint use:
        define_code_breakpoint,
        define_page_breakpoint,
        define_hardware_breakpoint,
        has_code_breakpoint,
        has_page_breakpoint,
        has_hardware_breakpoint,
        get_code_breakpoint,
        get_page_breakpoint,
        get_hardware_breakpoint,
        erase_code_breakpoint,
        erase_page_breakpoint,
        erase_hardware_breakpoint,
        enable_code_breakpoint,
        enable_page_breakpoint,
        enable_hardware_breakpoint,
        enable_one_shot_code_breakpoint,
        enable_one_shot_page_breakpoint,
        enable_one_shot_hardware_breakpoint,
        disable_code_breakpoint,
        disable_page_breakpoint,
        disable_hardware_breakpoint

    @group Listing breakpoints:
        get_all_breakpoints,
        get_all_code_breakpoints,
        get_all_page_breakpoints,
        get_all_hardware_breakpoints,
        get_process_breakpoints,
        get_process_code_breakpoints,
        get_process_page_breakpoints,
        get_process_hardware_breakpoints,
        get_thread_hardware_breakpoints,
        get_all_deferred_code_breakpoints,
        get_process_deferred_code_breakpoints

    @group Batch operations on breakpoints:
        enable_all_breakpoints,
        enable_one_shot_all_breakpoints,
        disable_all_breakpoints,
        erase_all_breakpoints,
        enable_process_breakpoints,
        enable_one_shot_process_breakpoints,
        disable_process_breakpoints,
        erase_process_breakpoints

    @group Breakpoint types:
        BP_TYPE_ANY, BP_TYPE_CODE, BP_TYPE_PAGE, BP_TYPE_HARDWARE
    @group Breakpoint states:
        BP_STATE_DISABLED, BP_STATE_ENABLED, BP_STATE_ONESHOT, BP_STATE_RUNNING
    @group Memory breakpoint trigger flags:
        BP_BREAK_ON_EXECUTION, BP_BREAK_ON_WRITE, BP_BREAK_ON_ACCESS
    @group Memory breakpoint size flags:
        BP_WATCH_BYTE, BP_WATCH_WORD, BP_WATCH_DWORD, BP_WATCH_QWORD

    @type BP_TYPE_ANY: int
    @cvar BP_TYPE_ANY: To get all breakpoints
    @type BP_TYPE_CODE: int
    @cvar BP_TYPE_CODE: To get code breakpoints only
    @type BP_TYPE_PAGE: int
    @cvar BP_TYPE_PAGE: To get page breakpoints only
    @type BP_TYPE_HARDWARE: int
    @cvar BP_TYPE_HARDWARE: To get hardware breakpoints only

    @type BP_STATE_DISABLED: int
    @cvar BP_STATE_DISABLED: Breakpoint is disabled.
    @type BP_STATE_ENABLED: int
    @cvar BP_STATE_ENABLED: Breakpoint is enabled.
    @type BP_STATE_ONESHOT: int
    @cvar BP_STATE_ONESHOT: Breakpoint is enabled for one shot.
    @type BP_STATE_RUNNING: int
    @cvar BP_STATE_RUNNING: Breakpoint is running (recently hit).

    @type BP_BREAK_ON_EXECUTION: int
    @cvar BP_BREAK_ON_EXECUTION: Break on code execution.
    @type BP_BREAK_ON_WRITE: int
    @cvar BP_BREAK_ON_WRITE: Break on memory write.
    @type BP_BREAK_ON_ACCESS: int
    @cvar BP_BREAK_ON_ACCESS: Break on memory read or write.
    """
    BP_TYPE_ANY = 0
    BP_TYPE_CODE = 1
    BP_TYPE_PAGE = 2
    BP_TYPE_HARDWARE = 3
    BP_STATE_DISABLED = Breakpoint.DISABLED
    BP_STATE_ENABLED = Breakpoint.ENABLED
    BP_STATE_ONESHOT = Breakpoint.ONESHOT
    BP_STATE_RUNNING = Breakpoint.RUNNING
    BP_BREAK_ON_EXECUTION = HardwareBreakpoint.BREAK_ON_EXECUTION
    BP_BREAK_ON_WRITE = HardwareBreakpoint.BREAK_ON_WRITE
    BP_BREAK_ON_ACCESS = HardwareBreakpoint.BREAK_ON_ACCESS
    BP_WATCH_BYTE = HardwareBreakpoint.WATCH_BYTE
    BP_WATCH_WORD = HardwareBreakpoint.WATCH_WORD
    BP_WATCH_QWORD = HardwareBreakpoint.WATCH_QWORD
    BP_WATCH_DWORD = HardwareBreakpoint.WATCH_DWORD

    def __init__(self):
        self.__codeBP = dict()
        self.__pageBP = dict()
        self.__hardwareBP = dict()
        self.__runningBP = dict()
        self.__tracing = set()
        self.__deferredBP = dict()

    def __get_running_bp_set(self, tid):
        """Auxiliary method."""
        return self.__runningBP.get(tid, ())

    def __add_running_bp(self, tid, bp):
        """Auxiliary method."""
        if tid not in self.__runningBP:
            self.__runningBP[tid] = set()
        self.__runningBP[tid].add(bp)

    def __del_running_bp(self, tid, bp):
        """Auxiliary method."""
        self.__runningBP[tid].remove(bp)
        if not self.__runningBP[tid]:
            del self.__runningBP[tid]

    def __del_running_bp_from_all_threads(self, bp):
        """Auxiliary method."""
        for tid, bpset in compat.iteritems(self.__runningBP):
            if bp in bpset:
                bpset.remove(bp)
                self.system.get_thread(tid).clear_tf()

    def __cleanup_breakpoint(self, event, bp):
        """Auxiliary method."""
        try:
            process = event.get_process()
            thread = event.get_thread()
            bp.disable(process, thread)
        except Exception:
            pass
        bp.set_condition(True)
        bp.set_action(None)

    def __cleanup_thread(self, event):
        """
        Auxiliary method for L{_notify_exit_thread}
        and L{_notify_exit_process}.
        """
        tid = event.get_tid()
        try:
            for bp in self.__runningBP[tid]:
                self.__cleanup_breakpoint(event, bp)
            del self.__runningBP[tid]
        except KeyError:
            pass
        try:
            for bp in self.__hardwareBP[tid]:
                self.__cleanup_breakpoint(event, bp)
            del self.__hardwareBP[tid]
        except KeyError:
            pass
        if tid in self.__tracing:
            self.__tracing.remove(tid)

    def __cleanup_process(self, event):
        """
        Auxiliary method for L{_notify_exit_process}.
        """
        pid = event.get_pid()
        process = event.get_process()
        for bp_pid, bp_address in compat.keys(self.__codeBP):
            if bp_pid == pid:
                bp = self.__codeBP[bp_pid, bp_address]
                self.__cleanup_breakpoint(event, bp)
                del self.__codeBP[bp_pid, bp_address]
        for bp_pid, bp_address in compat.keys(self.__pageBP):
            if bp_pid == pid:
                bp = self.__pageBP[bp_pid, bp_address]
                self.__cleanup_breakpoint(event, bp)
                del self.__pageBP[bp_pid, bp_address]
        try:
            del self.__deferredBP[pid]
        except KeyError:
            pass

    def __cleanup_module(self, event):
        """
        Auxiliary method for L{_notify_unload_dll}.
        """
        pid = event.get_pid()
        process = event.get_process()
        module = event.get_module()
        for tid in process.iter_thread_ids():
            thread = process.get_thread(tid)
            if tid in self.__runningBP:
                bplist = list(self.__runningBP[tid])
                for bp in bplist:
                    bp_address = bp.get_address()
                    if process.get_module_at_address(bp_address) == module:
                        self.__cleanup_breakpoint(event, bp)
                        self.__runningBP[tid].remove(bp)
            if tid in self.__hardwareBP:
                bplist = list(self.__hardwareBP[tid])
                for bp in bplist:
                    bp_address = bp.get_address()
                    if process.get_module_at_address(bp_address) == module:
                        self.__cleanup_breakpoint(event, bp)
                        self.__hardwareBP[tid].remove(bp)
        for bp_pid, bp_address in compat.keys(self.__codeBP):
            if bp_pid == pid:
                if process.get_module_at_address(bp_address) == module:
                    bp = self.__codeBP[bp_pid, bp_address]
                    self.__cleanup_breakpoint(event, bp)
                    del self.__codeBP[bp_pid, bp_address]
        for bp_pid, bp_address in compat.keys(self.__pageBP):
            if bp_pid == pid:
                if process.get_module_at_address(bp_address) == module:
                    bp = self.__pageBP[bp_pid, bp_address]
                    self.__cleanup_breakpoint(event, bp)
                    del self.__pageBP[bp_pid, bp_address]

    def define_code_breakpoint(self, dwProcessId, address, condition=True, action=None):
        """
        Creates a disabled code breakpoint at the given address.

        @see:
            L{has_code_breakpoint},
            L{get_code_breakpoint},
            L{enable_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint},
            L{erase_code_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of the code instruction to break at.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

            The callback signature is::

                def condition_callback(event):
                    return True     # returns True or False

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @type  action: function
        @param action: (Optional) Action callback function.
            If specified, the event is handled by this callback instead of
            being dispatched normally.

            The callback signature is::

                def action_callback(event):
                    pass        # no return value

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @rtype:  L{CodeBreakpoint}
        @return: The code breakpoint object.
        """
        process = self.system.get_process(dwProcessId)
        bp = CodeBreakpoint(address, condition, action)
        key = (dwProcessId, bp.get_address())
        if key in self.__codeBP:
            msg = 'Already exists (PID %d) : %r'
            raise KeyError(msg % (dwProcessId, self.__codeBP[key]))
        self.__codeBP[key] = bp
        return bp

    def define_page_breakpoint(self, dwProcessId, address, pages=1, condition=True, action=None):
        """
        Creates a disabled page breakpoint at the given address.

        @see:
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint},
            L{erase_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of the first page to watch.

        @type  pages: int
        @param pages: Number of pages to watch.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

            The callback signature is::

                def condition_callback(event):
                    return True     # returns True or False

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @type  action: function
        @param action: (Optional) Action callback function.
            If specified, the event is handled by this callback instead of
            being dispatched normally.

            The callback signature is::

                def action_callback(event):
                    pass        # no return value

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @rtype:  L{PageBreakpoint}
        @return: The page breakpoint object.
        """
        process = self.system.get_process(dwProcessId)
        bp = PageBreakpoint(address, pages, condition, action)
        begin = bp.get_address()
        end = begin + bp.get_size()
        address = begin
        pageSize = MemoryAddresses.pageSize
        while address < end:
            key = (dwProcessId, address)
            if key in self.__pageBP:
                msg = 'Already exists (PID %d) : %r'
                msg = msg % (dwProcessId, self.__pageBP[key])
                raise KeyError(msg)
            address = address + pageSize
        address = begin
        while address < end:
            key = (dwProcessId, address)
            self.__pageBP[key] = bp
            address = address + pageSize
        return bp

    def define_hardware_breakpoint(self, dwThreadId, address, triggerFlag=BP_BREAK_ON_ACCESS, sizeFlag=BP_WATCH_DWORD, condition=True, action=None):
        """
        Creates a disabled hardware breakpoint at the given address.

        @see:
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint},
            L{erase_hardware_breakpoint}

        @note:
            Hardware breakpoints do not seem to work properly on VirtualBox.
            See U{http://www.virtualbox.org/ticket/477}.

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address to watch.

        @type  triggerFlag: int
        @param triggerFlag: Trigger of breakpoint. Must be one of the following:

             - L{BP_BREAK_ON_EXECUTION}

               Break on code execution.

             - L{BP_BREAK_ON_WRITE}

               Break on memory read or write.

             - L{BP_BREAK_ON_ACCESS}

               Break on memory write.

        @type  sizeFlag: int
        @param sizeFlag: Size of breakpoint. Must be one of the following:

             - L{BP_WATCH_BYTE}

               One (1) byte in size.

             - L{BP_WATCH_WORD}

               Two (2) bytes in size.

             - L{BP_WATCH_DWORD}

               Four (4) bytes in size.

             - L{BP_WATCH_QWORD}

               Eight (8) bytes in size.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

            The callback signature is::

                def condition_callback(event):
                    return True     # returns True or False

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @type  action: function
        @param action: (Optional) Action callback function.
            If specified, the event is handled by this callback instead of
            being dispatched normally.

            The callback signature is::

                def action_callback(event):
                    pass        # no return value

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @rtype:  L{HardwareBreakpoint}
        @return: The hardware breakpoint object.
        """
        thread = self.system.get_thread(dwThreadId)
        bp = HardwareBreakpoint(address, triggerFlag, sizeFlag, condition, action)
        begin = bp.get_address()
        end = begin + bp.get_size()
        if dwThreadId in self.__hardwareBP:
            bpSet = self.__hardwareBP[dwThreadId]
            for oldbp in bpSet:
                old_begin = oldbp.get_address()
                old_end = old_begin + oldbp.get_size()
                if MemoryAddresses.do_ranges_intersect(begin, end, old_begin, old_end):
                    msg = 'Already exists (TID %d) : %r' % (dwThreadId, oldbp)
                    raise KeyError(msg)
        else:
            bpSet = set()
            self.__hardwareBP[dwThreadId] = bpSet
        bpSet.add(bp)
        return bp

    def has_code_breakpoint(self, dwProcessId, address):
        """
        Checks if a code breakpoint is defined at the given address.

        @see:
            L{define_code_breakpoint},
            L{get_code_breakpoint},
            L{erase_code_breakpoint},
            L{enable_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.

        @rtype:  bool
        @return: C{True} if the breakpoint is defined, C{False} otherwise.
        """
        return (dwProcessId, address) in self.__codeBP

    def has_page_breakpoint(self, dwProcessId, address):
        """
        Checks if a page breakpoint is defined at the given address.

        @see:
            L{define_page_breakpoint},
            L{get_page_breakpoint},
            L{erase_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.

        @rtype:  bool
        @return: C{True} if the breakpoint is defined, C{False} otherwise.
        """
        return (dwProcessId, address) in self.__pageBP

    def has_hardware_breakpoint(self, dwThreadId, address):
        """
        Checks if a hardware breakpoint is defined at the given address.

        @see:
            L{define_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{erase_hardware_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint}

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.

        @rtype:  bool
        @return: C{True} if the breakpoint is defined, C{False} otherwise.
        """
        if dwThreadId in self.__hardwareBP:
            bpSet = self.__hardwareBP[dwThreadId]
            for bp in bpSet:
                if bp.get_address() == address:
                    return True
        return False

    def get_code_breakpoint(self, dwProcessId, address):
        """
        Returns the internally used breakpoint object,
        for the code breakpoint defined at the given address.

        @warning: It's usually best to call the L{Debug} methods
            instead of accessing the breakpoint objects directly.

        @see:
            L{define_code_breakpoint},
            L{has_code_breakpoint},
            L{enable_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint},
            L{erase_code_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address where the breakpoint is defined.

        @rtype:  L{CodeBreakpoint}
        @return: The code breakpoint object.
        """
        key = (dwProcessId, address)
        if key not in self.__codeBP:
            msg = 'No breakpoint at process %d, address %s'
            address = HexDump.address(address)
            raise KeyError(msg % (dwProcessId, address))
        return self.__codeBP[key]

    def get_page_breakpoint(self, dwProcessId, address):
        """
        Returns the internally used breakpoint object,
        for the page breakpoint defined at the given address.

        @warning: It's usually best to call the L{Debug} methods
            instead of accessing the breakpoint objects directly.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint},
            L{erase_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address where the breakpoint is defined.

        @rtype:  L{PageBreakpoint}
        @return: The page breakpoint object.
        """
        key = (dwProcessId, address)
        if key not in self.__pageBP:
            msg = 'No breakpoint at process %d, address %s'
            address = HexDump.addresS(address)
            raise KeyError(msg % (dwProcessId, address))
        return self.__pageBP[key]

    def get_hardware_breakpoint(self, dwThreadId, address):
        """
        Returns the internally used breakpoint object,
        for the code breakpoint defined at the given address.

        @warning: It's usually best to call the L{Debug} methods
            instead of accessing the breakpoint objects directly.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_code_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint},
            L{erase_hardware_breakpoint}

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address where the breakpoint is defined.

        @rtype:  L{HardwareBreakpoint}
        @return: The hardware breakpoint object.
        """
        if dwThreadId not in self.__hardwareBP:
            msg = 'No hardware breakpoints set for thread %d'
            raise KeyError(msg % dwThreadId)
        for bp in self.__hardwareBP[dwThreadId]:
            if bp.is_here(address):
                return bp
        msg = 'No hardware breakpoint at thread %d, address %s'
        raise KeyError(msg % (dwThreadId, HexDump.address(address)))

    def enable_code_breakpoint(self, dwProcessId, address):
        """
        Enables the code breakpoint at the given address.

        @see:
            L{define_code_breakpoint},
            L{has_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint}
            L{erase_code_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        p = self.system.get_process(dwProcessId)
        bp = self.get_code_breakpoint(dwProcessId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.enable(p, None)

    def enable_page_breakpoint(self, dwProcessId, address):
        """
        Enables the page breakpoint at the given address.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint}
            L{erase_page_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        p = self.system.get_process(dwProcessId)
        bp = self.get_page_breakpoint(dwProcessId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.enable(p, None)

    def enable_hardware_breakpoint(self, dwThreadId, address):
        """
        Enables the hardware breakpoint at the given address.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint}
            L{erase_hardware_breakpoint},

        @note: Do not set hardware breakpoints while processing the system
            breakpoint event.

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        t = self.system.get_thread(dwThreadId)
        bp = self.get_hardware_breakpoint(dwThreadId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.enable(None, t)

    def enable_one_shot_code_breakpoint(self, dwProcessId, address):
        """
        Enables the code breakpoint at the given address for only one shot.

        @see:
            L{define_code_breakpoint},
            L{has_code_breakpoint},
            L{get_code_breakpoint},
            L{enable_code_breakpoint},
            L{disable_code_breakpoint}
            L{erase_code_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        p = self.system.get_process(dwProcessId)
        bp = self.get_code_breakpoint(dwProcessId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.one_shot(p, None)

    def enable_one_shot_page_breakpoint(self, dwProcessId, address):
        """
        Enables the page breakpoint at the given address for only one shot.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint},
            L{disable_page_breakpoint}
            L{erase_page_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        p = self.system.get_process(dwProcessId)
        bp = self.get_page_breakpoint(dwProcessId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.one_shot(p, None)

    def enable_one_shot_hardware_breakpoint(self, dwThreadId, address):
        """
        Enables the hardware breakpoint at the given address for only one shot.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_hardware_breakpoint},
            L{disable_hardware_breakpoint}
            L{erase_hardware_breakpoint},

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        t = self.system.get_thread(dwThreadId)
        bp = self.get_hardware_breakpoint(dwThreadId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.one_shot(None, t)

    def disable_code_breakpoint(self, dwProcessId, address):
        """
        Disables the code breakpoint at the given address.

        @see:
            L{define_code_breakpoint},
            L{has_code_breakpoint},
            L{get_code_breakpoint},
            L{enable_code_breakpoint}
            L{enable_one_shot_code_breakpoint},
            L{erase_code_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        p = self.system.get_process(dwProcessId)
        bp = self.get_code_breakpoint(dwProcessId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.disable(p, None)

    def disable_page_breakpoint(self, dwProcessId, address):
        """
        Disables the page breakpoint at the given address.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint}
            L{enable_one_shot_page_breakpoint},
            L{erase_page_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        p = self.system.get_process(dwProcessId)
        bp = self.get_page_breakpoint(dwProcessId, address)
        if bp.is_running():
            self.__del_running_bp_from_all_threads(bp)
        bp.disable(p, None)

    def disable_hardware_breakpoint(self, dwThreadId, address):
        """
        Disables the hardware breakpoint at the given address.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_hardware_breakpoint}
            L{enable_one_shot_hardware_breakpoint},
            L{erase_hardware_breakpoint},

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        t = self.system.get_thread(dwThreadId)
        p = t.get_process()
        bp = self.get_hardware_breakpoint(dwThreadId, address)
        if bp.is_running():
            self.__del_running_bp(dwThreadId, bp)
        bp.disable(p, t)

    def erase_code_breakpoint(self, dwProcessId, address):
        """
        Erases the code breakpoint at the given address.

        @see:
            L{define_code_breakpoint},
            L{has_code_breakpoint},
            L{get_code_breakpoint},
            L{enable_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        bp = self.get_code_breakpoint(dwProcessId, address)
        if not bp.is_disabled():
            self.disable_code_breakpoint(dwProcessId, address)
        del self.__codeBP[dwProcessId, address]

    def erase_page_breakpoint(self, dwProcessId, address):
        """
        Erases the page breakpoint at the given address.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        bp = self.get_page_breakpoint(dwProcessId, address)
        begin = bp.get_address()
        end = begin + bp.get_size()
        if not bp.is_disabled():
            self.disable_page_breakpoint(dwProcessId, address)
        address = begin
        pageSize = MemoryAddresses.pageSize
        while address < end:
            del self.__pageBP[dwProcessId, address]
            address = address + pageSize

    def erase_hardware_breakpoint(self, dwThreadId, address):
        """
        Erases the hardware breakpoint at the given address.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint}

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
        bp = self.get_hardware_breakpoint(dwThreadId, address)
        if not bp.is_disabled():
            self.disable_hardware_breakpoint(dwThreadId, address)
        bpSet = self.__hardwareBP[dwThreadId]
        bpSet.remove(bp)
        if not bpSet:
            del self.__hardwareBP[dwThreadId]

    def get_all_breakpoints(self):
        """
        Returns all breakpoint objects as a list of tuples.

        Each tuple contains:
         - Process global ID to which the breakpoint applies.
         - Thread global ID to which the breakpoint applies, or C{None}.
         - The L{Breakpoint} object itself.

        @note: If you're only interested in a specific breakpoint type, or in
            breakpoints for a specific process or thread, it's probably faster
            to call one of the following methods:
             - L{get_all_code_breakpoints}
             - L{get_all_page_breakpoints}
             - L{get_all_hardware_breakpoints}
             - L{get_process_code_breakpoints}
             - L{get_process_page_breakpoints}
             - L{get_process_hardware_breakpoints}
             - L{get_thread_hardware_breakpoints}

        @rtype:  list of tuple( pid, tid, bp )
        @return: List of all breakpoints.
        """
        bplist = list()
        for pid, bp in self.get_all_code_breakpoints():
            bplist.append((pid, None, bp))
        for pid, bp in self.get_all_page_breakpoints():
            bplist.append((pid, None, bp))
        for tid, bp in self.get_all_hardware_breakpoints():
            pid = self.system.get_thread(tid).get_pid()
            bplist.append((pid, tid, bp))
        return bplist

    def get_all_code_breakpoints(self):
        """
        @rtype:  list of tuple( int, L{CodeBreakpoint} )
        @return: All code breakpoints as a list of tuples (pid, bp).
        """
        return [(pid, bp) for (pid, address), bp in compat.iteritems(self.__codeBP)]

    def get_all_page_breakpoints(self):
        """
        @rtype:  list of tuple( int, L{PageBreakpoint} )
        @return: All page breakpoints as a list of tuples (pid, bp).
        """
        result = set()
        for (pid, address), bp in compat.iteritems(self.__pageBP):
            result.add((pid, bp))
        return list(result)

    def get_all_hardware_breakpoints(self):
        """
        @rtype:  list of tuple( int, L{HardwareBreakpoint} )
        @return: All hardware breakpoints as a list of tuples (tid, bp).
        """
        result = list()
        for tid, bplist in compat.iteritems(self.__hardwareBP):
            for bp in bplist:
                result.append((tid, bp))
        return result

    def get_process_breakpoints(self, dwProcessId):
        """
        Returns all breakpoint objects for the given process as a list of tuples.

        Each tuple contains:
         - Process global ID to which the breakpoint applies.
         - Thread global ID to which the breakpoint applies, or C{None}.
         - The L{Breakpoint} object itself.

        @note: If you're only interested in a specific breakpoint type, or in
            breakpoints for a specific process or thread, it's probably faster
            to call one of the following methods:
             - L{get_all_code_breakpoints}
             - L{get_all_page_breakpoints}
             - L{get_all_hardware_breakpoints}
             - L{get_process_code_breakpoints}
             - L{get_process_page_breakpoints}
             - L{get_process_hardware_breakpoints}
             - L{get_thread_hardware_breakpoints}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of tuple( pid, tid, bp )
        @return: List of all breakpoints for the given process.
        """
        bplist = list()
        for bp in self.get_process_code_breakpoints(dwProcessId):
            bplist.append((dwProcessId, None, bp))
        for bp in self.get_process_page_breakpoints(dwProcessId):
            bplist.append((dwProcessId, None, bp))
        for tid, bp in self.get_process_hardware_breakpoints(dwProcessId):
            pid = self.system.get_thread(tid).get_pid()
            bplist.append((dwProcessId, tid, bp))
        return bplist

    def get_process_code_breakpoints(self, dwProcessId):
        """
        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of L{CodeBreakpoint}
        @return: All code breakpoints for the given process.
        """
        return [bp for (pid, address), bp in compat.iteritems(self.__codeBP) if pid == dwProcessId]

    def get_process_page_breakpoints(self, dwProcessId):
        """
        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of L{PageBreakpoint}
        @return: All page breakpoints for the given process.
        """
        return [bp for (pid, address), bp in compat.iteritems(self.__pageBP) if pid == dwProcessId]

    def get_thread_hardware_breakpoints(self, dwThreadId):
        """
        @see: L{get_process_hardware_breakpoints}

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @rtype:  list of L{HardwareBreakpoint}
        @return: All hardware breakpoints for the given thread.
        """
        result = list()
        for tid, bplist in compat.iteritems(self.__hardwareBP):
            if tid == dwThreadId:
                for bp in bplist:
                    result.append(bp)
        return result

    def get_process_hardware_breakpoints(self, dwProcessId):
        """
        @see: L{get_thread_hardware_breakpoints}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of tuple( int, L{HardwareBreakpoint} )
        @return: All hardware breakpoints for each thread in the given process
            as a list of tuples (tid, bp).
        """
        result = list()
        aProcess = self.system.get_process(dwProcessId)
        for dwThreadId in aProcess.iter_thread_ids():
            if dwThreadId in self.__hardwareBP:
                bplist = self.__hardwareBP[dwThreadId]
                for bp in bplist:
                    result.append((dwThreadId, bp))
        return result

    def enable_all_breakpoints(self):
        """
        Enables all disabled breakpoints in all processes.

        @see:
            enable_code_breakpoint,
            enable_page_breakpoint,
            enable_hardware_breakpoint
        """
        for pid, bp in self.get_all_code_breakpoints():
            if bp.is_disabled():
                self.enable_code_breakpoint(pid, bp.get_address())
        for pid, bp in self.get_all_page_breakpoints():
            if bp.is_disabled():
                self.enable_page_breakpoint(pid, bp.get_address())
        for tid, bp in self.get_all_hardware_breakpoints():
            if bp.is_disabled():
                self.enable_hardware_breakpoint(tid, bp.get_address())

    def enable_one_shot_all_breakpoints(self):
        """
        Enables for one shot all disabled breakpoints in all processes.

        @see:
            enable_one_shot_code_breakpoint,
            enable_one_shot_page_breakpoint,
            enable_one_shot_hardware_breakpoint
        """
        for pid, bp in self.get_all_code_breakpoints():
            if bp.is_disabled():
                self.enable_one_shot_code_breakpoint(pid, bp.get_address())
        for pid, bp in self.get_all_page_breakpoints():
            if bp.is_disabled():
                self.enable_one_shot_page_breakpoint(pid, bp.get_address())
        for tid, bp in self.get_all_hardware_breakpoints():
            if bp.is_disabled():
                self.enable_one_shot_hardware_breakpoint(tid, bp.get_address())

    def disable_all_breakpoints(self):
        """
        Disables all breakpoints in all processes.

        @see:
            disable_code_breakpoint,
            disable_page_breakpoint,
            disable_hardware_breakpoint
        """
        for pid, bp in self.get_all_code_breakpoints():
            self.disable_code_breakpoint(pid, bp.get_address())
        for pid, bp in self.get_all_page_breakpoints():
            self.disable_page_breakpoint(pid, bp.get_address())
        for tid, bp in self.get_all_hardware_breakpoints():
            self.disable_hardware_breakpoint(tid, bp.get_address())

    def erase_all_breakpoints(self):
        """
        Erases all breakpoints in all processes.

        @see:
            erase_code_breakpoint,
            erase_page_breakpoint,
            erase_hardware_breakpoint
        """
        for pid, bp in self.get_all_code_breakpoints():
            self.erase_code_breakpoint(pid, bp.get_address())
        for pid, bp in self.get_all_page_breakpoints():
            self.erase_page_breakpoint(pid, bp.get_address())
        for tid, bp in self.get_all_hardware_breakpoints():
            self.erase_hardware_breakpoint(tid, bp.get_address())

    def enable_process_breakpoints(self, dwProcessId):
        """
        Enables all disabled breakpoints for the given process.

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.
        """
        for bp in self.get_process_code_breakpoints(dwProcessId):
            if bp.is_disabled():
                self.enable_code_breakpoint(dwProcessId, bp.get_address())
        for bp in self.get_process_page_breakpoints(dwProcessId):
            if bp.is_disabled():
                self.enable_page_breakpoint(dwProcessId, bp.get_address())
        if self.system.has_process(dwProcessId):
            aProcess = self.system.get_process(dwProcessId)
        else:
            aProcess = Process(dwProcessId)
            aProcess.scan_threads()
        for aThread in aProcess.iter_threads():
            dwThreadId = aThread.get_tid()
            for bp in self.get_thread_hardware_breakpoints(dwThreadId):
                if bp.is_disabled():
                    self.enable_hardware_breakpoint(dwThreadId, bp.get_address())

    def enable_one_shot_process_breakpoints(self, dwProcessId):
        """
        Enables for one shot all disabled breakpoints for the given process.

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.
        """
        for bp in self.get_process_code_breakpoints(dwProcessId):
            if bp.is_disabled():
                self.enable_one_shot_code_breakpoint(dwProcessId, bp.get_address())
        for bp in self.get_process_page_breakpoints(dwProcessId):
            if bp.is_disabled():
                self.enable_one_shot_page_breakpoint(dwProcessId, bp.get_address())
        if self.system.has_process(dwProcessId):
            aProcess = self.system.get_process(dwProcessId)
        else:
            aProcess = Process(dwProcessId)
            aProcess.scan_threads()
        for aThread in aProcess.iter_threads():
            dwThreadId = aThread.get_tid()
            for bp in self.get_thread_hardware_breakpoints(dwThreadId):
                if bp.is_disabled():
                    self.enable_one_shot_hardware_breakpoint(dwThreadId, bp.get_address())

    def disable_process_breakpoints(self, dwProcessId):
        """
        Disables all breakpoints for the given process.

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.
        """
        for bp in self.get_process_code_breakpoints(dwProcessId):
            self.disable_code_breakpoint(dwProcessId, bp.get_address())
        for bp in self.get_process_page_breakpoints(dwProcessId):
            self.disable_page_breakpoint(dwProcessId, bp.get_address())
        if self.system.has_process(dwProcessId):
            aProcess = self.system.get_process(dwProcessId)
        else:
            aProcess = Process(dwProcessId)
            aProcess.scan_threads()
        for aThread in aProcess.iter_threads():
            dwThreadId = aThread.get_tid()
            for bp in self.get_thread_hardware_breakpoints(dwThreadId):
                self.disable_hardware_breakpoint(dwThreadId, bp.get_address())

    def erase_process_breakpoints(self, dwProcessId):
        """
        Erases all breakpoints for the given process.

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.
        """
        self.disable_process_breakpoints(dwProcessId)
        for bp in self.get_process_code_breakpoints(dwProcessId):
            self.erase_code_breakpoint(dwProcessId, bp.get_address())
        for bp in self.get_process_page_breakpoints(dwProcessId):
            self.erase_page_breakpoint(dwProcessId, bp.get_address())
        if self.system.has_process(dwProcessId):
            aProcess = self.system.get_process(dwProcessId)
        else:
            aProcess = Process(dwProcessId)
            aProcess.scan_threads()
        for aThread in aProcess.iter_threads():
            dwThreadId = aThread.get_tid()
            for bp in self.get_thread_hardware_breakpoints(dwThreadId):
                self.erase_hardware_breakpoint(dwThreadId, bp.get_address())

    def _notify_guard_page(self, event):
        """
        Notify breakpoints of a guard page exception event.

        @type  event: L{ExceptionEvent}
        @param event: Guard page exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        address = event.get_fault_address()
        pid = event.get_pid()
        bCallHandler = True
        mask = ~(MemoryAddresses.pageSize - 1)
        address = address & mask
        key = (pid, address)
        if key in self.__pageBP:
            bp = self.__pageBP[key]
            if bp.is_enabled() or bp.is_one_shot():
                event.continueStatus = win32.DBG_CONTINUE
                bp.hit(event)
                if bp.is_running():
                    tid = event.get_tid()
                    self.__add_running_bp(tid, bp)
                bCondition = bp.eval_condition(event)
                if bCondition and bp.is_automatic():
                    bp.run_action(event)
                    bCallHandler = False
                else:
                    bCallHandler = bCondition
        else:
            event.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
        return bCallHandler

    def _notify_breakpoint(self, event):
        """
        Notify breakpoints of a breakpoint exception event.

        @type  event: L{ExceptionEvent}
        @param event: Breakpoint exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        address = event.get_exception_address()
        pid = event.get_pid()
        bCallHandler = True
        key = (pid, address)
        if key in self.__codeBP:
            bp = self.__codeBP[key]
            if not bp.is_disabled():
                aThread = event.get_thread()
                aThread.set_pc(address)
                event.continueStatus = win32.DBG_CONTINUE
                bp.hit(event)
                if bp.is_running():
                    tid = event.get_tid()
                    self.__add_running_bp(tid, bp)
                bCondition = bp.eval_condition(event)
                if bCondition and bp.is_automatic():
                    bCallHandler = bp.run_action(event)
                else:
                    bCallHandler = bCondition
        elif event.get_process().is_system_defined_breakpoint(address):
            event.continueStatus = win32.DBG_CONTINUE
        elif self.in_hostile_mode():
            event.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
        else:
            event.continueStatus = win32.DBG_CONTINUE
        return bCallHandler

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

    def _notify_load_dll(self, event):
        """
        Notify the loading of a DLL.

        @type  event: L{LoadDLLEvent}
        @param event: Load DLL event.

        @rtype:  bool
        @return: C{True} to call the user-defined handler, C{False} otherwise.
        """
        self.__set_deferred_breakpoints(event)
        return True

    def _notify_unload_dll(self, event):
        """
        Notify the unloading of a DLL.

        @type  event: L{UnloadDLLEvent}
        @param event: Unload DLL event.

        @rtype:  bool
        @return: C{True} to call the user-defined handler, C{False} otherwise.
        """
        self.__cleanup_module(event)
        return True

    def _notify_exit_thread(self, event):
        """
        Notify the termination of a thread.

        @type  event: L{ExitThreadEvent}
        @param event: Exit thread event.

        @rtype:  bool
        @return: C{True} to call the user-defined handler, C{False} otherwise.
        """
        self.__cleanup_thread(event)
        return True

    def _notify_exit_process(self, event):
        """
        Notify the termination of a process.

        @type  event: L{ExitProcessEvent}
        @param event: Exit process event.

        @rtype:  bool
        @return: C{True} to call the user-defined handler, C{False} otherwise.
        """
        self.__cleanup_process(event)
        self.__cleanup_thread(event)
        return True

    def __set_break(self, pid, address, action, oneshot):
        """
        Used by L{break_at} and L{stalk_at}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_code_breakpoint} for more details.

        @type  oneshot: bool
        @param oneshot: C{True} for one-shot breakpoints, C{False} otherwise.

        @rtype:  L{Breakpoint}
        @return: Returns the new L{Breakpoint} object, or C{None} if the label
            couldn't be resolved and the breakpoint was deferred. Deferred
            breakpoints are set when the DLL they point to is loaded.
        """
        if type(address) not in (int, long):
            label = address
            try:
                address = self.system.get_process(pid).resolve_label(address)
                if not address:
                    raise Exception()
            except Exception:
                try:
                    deferred = self.__deferredBP[pid]
                except KeyError:
                    deferred = dict()
                    self.__deferredBP[pid] = deferred
                if label in deferred:
                    msg = 'Redefined deferred code breakpoint at %s in process ID %d'
                    msg = msg % (label, pid)
                    warnings.warn(msg, BreakpointWarning)
                deferred[label] = (action, oneshot)
                return None
        if self.has_code_breakpoint(pid, address):
            bp = self.get_code_breakpoint(pid, address)
            if bp.get_action() != action:
                bp.set_action(action)
                msg = 'Redefined code breakpoint at %s in process ID %d'
                msg = msg % (label, pid)
                warnings.warn(msg, BreakpointWarning)
        else:
            self.define_code_breakpoint(pid, address, True, action)
            bp = self.get_code_breakpoint(pid, address)
        if oneshot:
            if not bp.is_one_shot():
                self.enable_one_shot_code_breakpoint(pid, address)
        elif not bp.is_enabled():
            self.enable_code_breakpoint(pid, address)
        return bp

    def __clear_break(self, pid, address):
        """
        Used by L{dont_break_at} and L{dont_stalk_at}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.
        """
        if type(address) not in (int, long):
            unknown = True
            label = address
            try:
                deferred = self.__deferredBP[pid]
                del deferred[label]
                unknown = False
            except KeyError:
                pass
            aProcess = self.system.get_process(pid)
            try:
                address = aProcess.resolve_label(label)
                if not address:
                    raise Exception()
            except Exception:
                if unknown:
                    msg = "Can't clear unknown code breakpoint at %s in process ID %d"
                    msg = msg % (label, pid)
                    warnings.warn(msg, BreakpointWarning)
                return
        if self.has_code_breakpoint(pid, address):
            self.erase_code_breakpoint(pid, address)

    def __set_deferred_breakpoints(self, event):
        """
        Used internally. Sets all deferred breakpoints for a DLL when it's
        loaded.

        @type  event: L{LoadDLLEvent}
        @param event: Load DLL event.
        """
        pid = event.get_pid()
        try:
            deferred = self.__deferredBP[pid]
        except KeyError:
            return
        aProcess = event.get_process()
        for label, (action, oneshot) in deferred.items():
            try:
                address = aProcess.resolve_label(label)
            except Exception:
                continue
            del deferred[label]
            try:
                self.__set_break(pid, address, action, oneshot)
            except Exception:
                msg = "Can't set deferred breakpoint %s at process ID %d"
                msg = msg % (label, pid)
                warnings.warn(msg, BreakpointWarning)

    def get_all_deferred_code_breakpoints(self):
        """
        Returns a list of deferred code breakpoints.

        @rtype:  tuple of (int, str, callable, bool)
        @return: Tuple containing the following elements:
             - Process ID where to set the breakpoint.
             - Label pointing to the address where to set the breakpoint.
             - Action callback for the breakpoint.
             - C{True} of the breakpoint is one-shot, C{False} otherwise.
        """
        result = []
        for pid, deferred in compat.iteritems(self.__deferredBP):
            for label, (action, oneshot) in compat.iteritems(deferred):
                result.add((pid, label, action, oneshot))
        return result

    def get_process_deferred_code_breakpoints(self, dwProcessId):
        """
        Returns a list of deferred code breakpoints.

        @type  dwProcessId: int
        @param dwProcessId: Process ID.

        @rtype:  tuple of (int, str, callable, bool)
        @return: Tuple containing the following elements:
             - Label pointing to the address where to set the breakpoint.
             - Action callback for the breakpoint.
             - C{True} of the breakpoint is one-shot, C{False} otherwise.
        """
        return [(label, action, oneshot) for label, (action, oneshot) in compat.iteritems(self.__deferredBP.get(dwProcessId, {}))]

    def stalk_at(self, pid, address, action=None):
        """
        Sets a one shot code breakpoint at the given process and address.

        If instead of an address you pass a label, the breakpoint may be
        deferred until the DLL it points to is loaded.

        @see: L{break_at}, L{dont_stalk_at}

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_code_breakpoint} for more details.

        @rtype:  bool
        @return: C{True} if the breakpoint was set immediately, or C{False} if
            it was deferred.
        """
        bp = self.__set_break(pid, address, action, oneshot=True)
        return bp is not None

    def break_at(self, pid, address, action=None):
        """
        Sets a code breakpoint at the given process and address.

        If instead of an address you pass a label, the breakpoint may be
        deferred until the DLL it points to is loaded.

        @see: L{stalk_at}, L{dont_break_at}

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_code_breakpoint} for more details.

        @rtype:  bool
        @return: C{True} if the breakpoint was set immediately, or C{False} if
            it was deferred.
        """
        bp = self.__set_break(pid, address, action, oneshot=False)
        return bp is not None

    def dont_break_at(self, pid, address):
        """
        Clears a code breakpoint set by L{break_at}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.
        """
        self.__clear_break(pid, address)

    def dont_stalk_at(self, pid, address):
        """
        Clears a code breakpoint set by L{stalk_at}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.
        """
        self.__clear_break(pid, address)

    def hook_function(self, pid, address, preCB=None, postCB=None, paramCount=None, signature=None):
        """
        Sets a function hook at the given address.

        If instead of an address you pass a label, the hook may be
        deferred until the DLL it points to is loaded.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  preCB: function
        @param preCB: (Optional) Callback triggered on function entry.

            The signature for the callback should be something like this::

                def pre_LoadLibraryEx(event, ra, lpFilename, hFile, dwFlags):

                    # return address
                    ra = params[0]

                    # function arguments start from here...
                    szFilename = event.get_process().peek_string(lpFilename)

                    # (...)

            Note that all pointer types are treated like void pointers, so your
            callback won't get the string or structure pointed to by it, but
            the remote memory address instead. This is so to prevent the ctypes
            library from being "too helpful" and trying to dereference the
            pointer. To get the actual data being pointed to, use one of the
            L{Process.read} methods.

        @type  postCB: function
        @param postCB: (Optional) Callback triggered on function exit.

            The signature for the callback should be something like this::

                def post_LoadLibraryEx(event, return_value):

                    # (...)

        @type  paramCount: int
        @param paramCount:
            (Optional) Number of parameters for the C{preCB} callback,
            not counting the return address. Parameters are read from
            the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

            This is a faster way to pull stack parameters in 32 bits, but in 64
            bits (or with some odd APIs in 32 bits) it won't be useful, since
            not all arguments to the hooked function will be of the same size.

            For a more reliable and cross-platform way of hooking use the
            C{signature} argument instead.

        @type  signature: tuple
        @param signature:
            (Optional) Tuple of C{ctypes} data types that constitute the
            hooked function signature. When the function is called, this will
            be used to parse the arguments from the stack. Overrides the
            C{paramCount} argument.

        @rtype:  bool
        @return: C{True} if the hook was set immediately, or C{False} if
            it was deferred.
        """
        try:
            aProcess = self.system.get_process(pid)
        except KeyError:
            aProcess = Process(pid)
        arch = aProcess.get_arch()
        hookObj = Hook(preCB, postCB, paramCount, signature, arch)
        bp = self.break_at(pid, address, hookObj)
        return bp is not None

    def stalk_function(self, pid, address, preCB=None, postCB=None, paramCount=None, signature=None):
        """
        Sets a one-shot function hook at the given address.

        If instead of an address you pass a label, the hook may be
        deferred until the DLL it points to is loaded.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  preCB: function
        @param preCB: (Optional) Callback triggered on function entry.

            The signature for the callback should be something like this::

                def pre_LoadLibraryEx(event, ra, lpFilename, hFile, dwFlags):

                    # return address
                    ra = params[0]

                    # function arguments start from here...
                    szFilename = event.get_process().peek_string(lpFilename)

                    # (...)

            Note that all pointer types are treated like void pointers, so your
            callback won't get the string or structure pointed to by it, but
            the remote memory address instead. This is so to prevent the ctypes
            library from being "too helpful" and trying to dereference the
            pointer. To get the actual data being pointed to, use one of the
            L{Process.read} methods.

        @type  postCB: function
        @param postCB: (Optional) Callback triggered on function exit.

            The signature for the callback should be something like this::

                def post_LoadLibraryEx(event, return_value):

                    # (...)

        @type  paramCount: int
        @param paramCount:
            (Optional) Number of parameters for the C{preCB} callback,
            not counting the return address. Parameters are read from
            the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

            This is a faster way to pull stack parameters in 32 bits, but in 64
            bits (or with some odd APIs in 32 bits) it won't be useful, since
            not all arguments to the hooked function will be of the same size.

            For a more reliable and cross-platform way of hooking use the
            C{signature} argument instead.

        @type  signature: tuple
        @param signature:
            (Optional) Tuple of C{ctypes} data types that constitute the
            hooked function signature. When the function is called, this will
            be used to parse the arguments from the stack. Overrides the
            C{paramCount} argument.

        @rtype:  bool
        @return: C{True} if the breakpoint was set immediately, or C{False} if
            it was deferred.
        """
        try:
            aProcess = self.system.get_process(pid)
        except KeyError:
            aProcess = Process(pid)
        arch = aProcess.get_arch()
        hookObj = Hook(preCB, postCB, paramCount, signature, arch)
        bp = self.stalk_at(pid, address, hookObj)
        return bp is not None

    def dont_hook_function(self, pid, address):
        """
        Removes a function hook set by L{hook_function}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.
        """
        self.dont_break_at(pid, address)
    unhook_function = dont_hook_function

    def dont_stalk_function(self, pid, address):
        """
        Removes a function hook set by L{stalk_function}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.
        """
        self.dont_stalk_at(pid, address)

    def __set_variable_watch(self, tid, address, size, action):
        """
        Used by L{watch_variable} and L{stalk_variable}.

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to watch.

        @type  size: int
        @param size: Size of variable to watch. The only supported sizes are:
            byte (1), word (2), dword (4) and qword (8).

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_hardware_breakpoint} for more details.

        @rtype:  L{HardwareBreakpoint}
        @return: Hardware breakpoint at the requested address.
        """
        if size == 1:
            sizeFlag = self.BP_WATCH_BYTE
        elif size == 2:
            sizeFlag = self.BP_WATCH_WORD
        elif size == 4:
            sizeFlag = self.BP_WATCH_DWORD
        elif size == 8:
            sizeFlag = self.BP_WATCH_QWORD
        else:
            raise ValueError('Bad size for variable watch: %r' % size)
        if self.has_hardware_breakpoint(tid, address):
            warnings.warn('Hardware breakpoint in thread %d at address %s was overwritten!' % (tid, HexDump.address(address, self.system.get_thread(tid).get_bits())), BreakpointWarning)
            bp = self.get_hardware_breakpoint(tid, address)
            if bp.get_trigger() != self.BP_BREAK_ON_ACCESS or bp.get_watch() != sizeFlag:
                self.erase_hardware_breakpoint(tid, address)
                self.define_hardware_breakpoint(tid, address, self.BP_BREAK_ON_ACCESS, sizeFlag, True, action)
                bp = self.get_hardware_breakpoint(tid, address)
        else:
            self.define_hardware_breakpoint(tid, address, self.BP_BREAK_ON_ACCESS, sizeFlag, True, action)
            bp = self.get_hardware_breakpoint(tid, address)
        return bp

    def __clear_variable_watch(self, tid, address):
        """
        Used by L{dont_watch_variable} and L{dont_stalk_variable}.

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to stop watching.
        """
        if self.has_hardware_breakpoint(tid, address):
            self.erase_hardware_breakpoint(tid, address)

    def watch_variable(self, tid, address, size, action=None):
        """
        Sets a hardware breakpoint at the given thread, address and size.

        @see: L{dont_watch_variable}

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to watch.

        @type  size: int
        @param size: Size of variable to watch. The only supported sizes are:
            byte (1), word (2), dword (4) and qword (8).

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_hardware_breakpoint} for more details.
        """
        bp = self.__set_variable_watch(tid, address, size, action)
        if not bp.is_enabled():
            self.enable_hardware_breakpoint(tid, address)

    def stalk_variable(self, tid, address, size, action=None):
        """
        Sets a one-shot hardware breakpoint at the given thread,
        address and size.

        @see: L{dont_watch_variable}

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to watch.

        @type  size: int
        @param size: Size of variable to watch. The only supported sizes are:
            byte (1), word (2), dword (4) and qword (8).

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_hardware_breakpoint} for more details.
        """
        bp = self.__set_variable_watch(tid, address, size, action)
        if not bp.is_one_shot():
            self.enable_one_shot_hardware_breakpoint(tid, address)

    def dont_watch_variable(self, tid, address):
        """
        Clears a hardware breakpoint set by L{watch_variable}.

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to stop watching.
        """
        self.__clear_variable_watch(tid, address)

    def dont_stalk_variable(self, tid, address):
        """
        Clears a hardware breakpoint set by L{stalk_variable}.

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to stop watching.
        """
        self.__clear_variable_watch(tid, address)

    def __set_buffer_watch(self, pid, address, size, action, bOneShot):
        """
        Used by L{watch_buffer} and L{stalk_buffer}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int
        @param address: Memory address of buffer to watch.

        @type  size: int
        @param size: Size in bytes of buffer to watch.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_page_breakpoint} for more details.

        @type  bOneShot: bool
        @param bOneShot:
            C{True} to set a one-shot breakpoint,
            C{False} to set a normal breakpoint.
        """
        if size < 1:
            raise ValueError('Bad size for buffer watch: %r' % size)
        bw = BufferWatch(pid, address, address + size, action, bOneShot)
        base = MemoryAddresses.align_address_to_page_start(address)
        limit = MemoryAddresses.align_address_to_page_end(address + size)
        pages = MemoryAddresses.get_buffer_size_in_pages(address, size)
        try:
            bset = set()
            nset = set()
            cset = set()
            page_addr = base
            pageSize = MemoryAddresses.pageSize
            while page_addr < limit:
                if self.has_page_breakpoint(pid, page_addr):
                    bp = self.get_page_breakpoint(pid, page_addr)
                    if bp not in bset:
                        condition = bp.get_condition()
                        if not condition in cset:
                            if not isinstance(condition, _BufferWatchCondition):
                                msg = "Can't watch buffer at page %s"
                                msg = msg % HexDump.address(page_addr)
                                raise RuntimeError(msg)
                            cset.add(condition)
                        bset.add(bp)
                else:
                    condition = _BufferWatchCondition()
                    bp = self.define_page_breakpoint(pid, page_addr, 1, condition=condition)
                    bset.add(bp)
                    nset.add(bp)
                    cset.add(condition)
                page_addr = page_addr + pageSize
            aProcess = self.system.get_process(pid)
            for bp in bset:
                if bp.is_disabled() or bp.is_one_shot():
                    bp.enable(aProcess, None)
        except:
            for bp in nset:
                try:
                    self.erase_page_breakpoint(pid, bp.get_address())
                except:
                    pass
            raise
        for condition in cset:
            condition.add(bw)

    def __clear_buffer_watch_old_method(self, pid, address, size):
        """
        Used by L{dont_watch_buffer} and L{dont_stalk_buffer}.

        @warn: Deprecated since WinAppDbg 1.5.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int
        @param address: Memory address of buffer to stop watching.

        @type  size: int
        @param size: Size in bytes of buffer to stop watching.
        """
        warnings.warn('Deprecated since WinAppDbg 1.5', DeprecationWarning)
        if size < 1:
            raise ValueError('Bad size for buffer watch: %r' % size)
        base = MemoryAddresses.align_address_to_page_start(address)
        limit = MemoryAddresses.align_address_to_page_end(address + size)
        pages = MemoryAddresses.get_buffer_size_in_pages(address, size)
        cset = set()
        page_addr = base
        pageSize = MemoryAddresses.pageSize
        while page_addr < limit:
            if self.has_page_breakpoint(pid, page_addr):
                bp = self.get_page_breakpoint(pid, page_addr)
                condition = bp.get_condition()
                if condition not in cset:
                    if not isinstance(condition, _BufferWatchCondition):
                        continue
                    cset.add(condition)
                    condition.remove_last_match(address, size)
                    if condition.count() == 0:
                        try:
                            self.erase_page_breakpoint(pid, bp.get_address())
                        except WindowsError:
                            pass
            page_addr = page_addr + pageSize

    def __clear_buffer_watch(self, bw):
        """
        Used by L{dont_watch_buffer} and L{dont_stalk_buffer}.

        @type  bw: L{BufferWatch}
        @param bw: Buffer watch identifier.
        """
        pid = bw.pid
        start = bw.start
        end = bw.end
        base = MemoryAddresses.align_address_to_page_start(start)
        limit = MemoryAddresses.align_address_to_page_end(end)
        pages = MemoryAddresses.get_buffer_size_in_pages(start, end - start)
        cset = set()
        page_addr = base
        pageSize = MemoryAddresses.pageSize
        while page_addr < limit:
            if self.has_page_breakpoint(pid, page_addr):
                bp = self.get_page_breakpoint(pid, page_addr)
                condition = bp.get_condition()
                if condition not in cset:
                    if not isinstance(condition, _BufferWatchCondition):
                        continue
                    cset.add(condition)
                    condition.remove(bw)
                    if condition.count() == 0:
                        try:
                            self.erase_page_breakpoint(pid, bp.get_address())
                        except WindowsError:
                            msg = 'Cannot remove page breakpoint at address %s'
                            msg = msg % HexDump.address(bp.get_address())
                            warnings.warn(msg, BreakpointWarning)
            page_addr = page_addr + pageSize

    def watch_buffer(self, pid, address, size, action=None):
        """
        Sets a page breakpoint and notifies when the given buffer is accessed.

        @see: L{dont_watch_variable}

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int
        @param address: Memory address of buffer to watch.

        @type  size: int
        @param size: Size in bytes of buffer to watch.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_page_breakpoint} for more details.

        @rtype:  L{BufferWatch}
        @return: Buffer watch identifier.
        """
        self.__set_buffer_watch(pid, address, size, action, False)

    def stalk_buffer(self, pid, address, size, action=None):
        """
        Sets a one-shot page breakpoint and notifies
        when the given buffer is accessed.

        @see: L{dont_watch_variable}

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int
        @param address: Memory address of buffer to watch.

        @type  size: int
        @param size: Size in bytes of buffer to watch.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_page_breakpoint} for more details.

        @rtype:  L{BufferWatch}
        @return: Buffer watch identifier.
        """
        self.__set_buffer_watch(pid, address, size, action, True)

    def dont_watch_buffer(self, bw, *argv, **argd):
        """
        Clears a page breakpoint set by L{watch_buffer}.

        @type  bw: L{BufferWatch}
        @param bw:
            Buffer watch identifier returned by L{watch_buffer}.
        """
        if not (argv or argd):
            self.__clear_buffer_watch(bw)
        else:
            argv = list(argv)
            argv.insert(0, bw)
            if 'pid' in argd:
                argv.insert(0, argd.pop('pid'))
            if 'address' in argd:
                argv.insert(1, argd.pop('address'))
            if 'size' in argd:
                argv.insert(2, argd.pop('size'))
            if argd:
                raise TypeError('Wrong arguments for dont_watch_buffer()')
            try:
                pid, address, size = argv
            except ValueError:
                raise TypeError('Wrong arguments for dont_watch_buffer()')
            self.__clear_buffer_watch_old_method(pid, address, size)

    def dont_stalk_buffer(self, bw, *argv, **argd):
        """
        Clears a page breakpoint set by L{stalk_buffer}.

        @type  bw: L{BufferWatch}
        @param bw:
            Buffer watch identifier returned by L{stalk_buffer}.
        """
        self.dont_watch_buffer(bw, *argv, **argd)

    def __start_tracing(self, thread):
        """
        @type  thread: L{Thread}
        @param thread: Thread to start tracing.
        """
        tid = thread.get_tid()
        if not tid in self.__tracing:
            thread.set_tf()
            self.__tracing.add(tid)

    def __stop_tracing(self, thread):
        """
        @type  thread: L{Thread}
        @param thread: Thread to stop tracing.
        """
        tid = thread.get_tid()
        if tid in self.__tracing:
            self.__tracing.remove(tid)
            if thread.is_alive():
                thread.clear_tf()

    def is_tracing(self, tid):
        """
        @type  tid: int
        @param tid: Thread global ID.

        @rtype:  bool
        @return: C{True} if the thread is being traced, C{False} otherwise.
        """
        return tid in self.__tracing

    def get_traced_tids(self):
        """
        Retrieves the list of global IDs of all threads being traced.

        @rtype:  list( int... )
        @return: List of thread global IDs.
        """
        tids = list(self.__tracing)
        tids.sort()
        return tids

    def start_tracing(self, tid):
        """
        Start tracing mode in the given thread.

        @type  tid: int
        @param tid: Global ID of thread to start tracing.
        """
        if not self.is_tracing(tid):
            thread = self.system.get_thread(tid)
            self.__start_tracing(thread)

    def stop_tracing(self, tid):
        """
        Stop tracing mode in the given thread.

        @type  tid: int
        @param tid: Global ID of thread to stop tracing.
        """
        if self.is_tracing(tid):
            thread = self.system.get_thread(tid)
            self.__stop_tracing(thread)

    def start_tracing_process(self, pid):
        """
        Start tracing mode for all threads in the given process.

        @type  pid: int
        @param pid: Global ID of process to start tracing.
        """
        for thread in self.system.get_process(pid).iter_threads():
            self.__start_tracing(thread)

    def stop_tracing_process(self, pid):
        """
        Stop tracing mode for all threads in the given process.

        @type  pid: int
        @param pid: Global ID of process to stop tracing.
        """
        for thread in self.system.get_process(pid).iter_threads():
            self.__stop_tracing(thread)

    def start_tracing_all(self):
        """
        Start tracing mode for all threads in all debugees.
        """
        for pid in self.get_debugee_pids():
            self.start_tracing_process(pid)

    def stop_tracing_all(self):
        """
        Stop tracing mode for all threads in all debugees.
        """
        for pid in self.get_debugee_pids():
            self.stop_tracing_process(pid)

    def break_on_error(self, pid, errorCode):
        """
        Sets or clears the system breakpoint for a given Win32 error code.

        Use L{Process.is_system_defined_breakpoint} to tell if a breakpoint
        exception was caused by a system breakpoint or by the application
        itself (for example because of a failed assertion in the code).

        @note: This functionality is only available since Windows Server 2003.
            In 2003 it only breaks on error values set externally to the
            kernel32.dll library, but this was fixed in Windows Vista.

        @warn: This method will fail if the debug symbols for ntdll (kernel32
            in Windows 2003) are not present. For more information see:
            L{System.fix_symbol_store_path}.

        @see: U{http://www.nynaeve.net/?p=147}

        @type  pid: int
        @param pid: Process ID.

        @type  errorCode: int
        @param errorCode: Win32 error code to stop on. Set to C{0} or
            C{ERROR_SUCCESS} to clear the breakpoint instead.

        @raise NotImplementedError:
            The functionality is not supported in this system.

        @raise WindowsError:
            An error occurred while processing this request.
        """
        aProcess = self.system.get_process(pid)
        address = aProcess.get_break_on_error_ptr()
        if not address:
            raise NotImplementedError('The functionality is not supported in this system.')
        aProcess.write_dword(address, errorCode)

    def dont_break_on_error(self, pid):
        """
        Alias to L{break_on_error}C{(pid, ERROR_SUCCESS)}.

        @type  pid: int
        @param pid: Process ID.

        @raise NotImplementedError:
            The functionality is not supported in this system.

        @raise WindowsError:
            An error occurred while processing this request.
        """
        self.break_on_error(pid, 0)

    def resolve_exported_function(self, pid, modName, procName):
        """
        Resolves the exported DLL function for the given process.

        @type  pid: int
        @param pid: Process global ID.

        @type  modName: str
        @param modName: Name of the module that exports the function.

        @type  procName: str
        @param procName: Name of the exported function to resolve.

        @rtype:  int, None
        @return: On success, the address of the exported function.
            On failure, returns C{None}.
        """
        aProcess = self.system.get_process(pid)
        aModule = aProcess.get_module_by_name(modName)
        if not aModule:
            aProcess.scan_modules()
            aModule = aProcess.get_module_by_name(modName)
        if aModule:
            address = aModule.resolve(procName)
            return address
        return None

    def resolve_label(self, pid, label):
        """
        Resolves a label for the given process.

        @type  pid: int
        @param pid: Process global ID.

        @type  label: str
        @param label: Label to resolve.

        @rtype:  int
        @return: Memory address pointed to by the label.

        @raise ValueError: The label is malformed or impossible to resolve.
        @raise RuntimeError: Cannot resolve the module or function.
        """
        return self.get_process(pid).resolve_label(label)