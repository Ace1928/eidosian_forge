from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
class _ProcessContainer(object):
    """
    Encapsulates the capability to contain Process objects.

    @group Instrumentation:
        start_process, argv_to_cmdline, cmdline_to_argv, get_explorer_pid

    @group Processes snapshot:
        scan, scan_processes, scan_processes_fast,
        get_process, get_process_count, get_process_ids,
        has_process, iter_processes, iter_process_ids,
        find_processes_by_filename, get_pid_from_tid,
        get_windows,
        scan_process_filenames,
        clear, clear_processes, clear_dead_processes,
        clear_unattached_processes,
        close_process_handles,
        close_process_and_thread_handles

    @group Threads snapshots:
        scan_processes_and_threads,
        get_thread, get_thread_count, get_thread_ids,
        has_thread

    @group Modules snapshots:
        scan_modules, find_modules_by_address,
        find_modules_by_base, find_modules_by_name,
        get_module_count
    """

    def __init__(self):
        self.__processDict = dict()

    def __initialize_snapshot(self):
        """
        Private method to automatically initialize the snapshot
        when you try to use it without calling any of the scan_*
        methods first. You don't need to call this yourself.
        """
        if not self.__processDict:
            try:
                self.scan_processes()
            except Exception:
                self.scan_processes_fast()
            self.scan_process_filenames()

    def __contains__(self, anObject):
        """
        @type  anObject: L{Process}, L{Thread}, int
        @param anObject:
             - C{int}: Global ID of the process to look for.
             - C{int}: Global ID of the thread to look for.
             - C{Process}: Process object to look for.
             - C{Thread}: Thread object to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains
            a L{Process} or L{Thread} object with the same ID.
        """
        if isinstance(anObject, Process):
            anObject = anObject.dwProcessId
        if self.has_process(anObject):
            return True
        for aProcess in self.iter_processes():
            if anObject in aProcess:
                return True
        return False

    def __iter__(self):
        """
        @see:    L{iter_processes}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Process} objects in this snapshot.
        """
        return self.iter_processes()

    def __len__(self):
        """
        @see:    L{get_process_count}
        @rtype:  int
        @return: Count of L{Process} objects in this snapshot.
        """
        return self.get_process_count()

    def has_process(self, dwProcessId):
        """
        @type  dwProcessId: int
        @param dwProcessId: Global ID of the process to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains a
            L{Process} object with the given global ID.
        """
        self.__initialize_snapshot()
        return dwProcessId in self.__processDict

    def get_process(self, dwProcessId):
        """
        @type  dwProcessId: int
        @param dwProcessId: Global ID of the process to look for.

        @rtype:  L{Process}
        @return: Process object with the given global ID.
        """
        self.__initialize_snapshot()
        if dwProcessId not in self.__processDict:
            msg = 'Unknown process ID %d' % dwProcessId
            raise KeyError(msg)
        return self.__processDict[dwProcessId]

    def iter_process_ids(self):
        """
        @see:    L{iter_processes}
        @rtype:  dictionary-keyiterator
        @return: Iterator of global process IDs in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.iterkeys(self.__processDict)

    def iter_processes(self):
        """
        @see:    L{iter_process_ids}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Process} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.itervalues(self.__processDict)

    def get_process_ids(self):
        """
        @see:    L{iter_process_ids}
        @rtype:  list( int )
        @return: List of global process IDs in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.keys(self.__processDict)

    def get_process_count(self):
        """
        @rtype:  int
        @return: Count of L{Process} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return len(self.__processDict)

    def get_windows(self):
        """
        @rtype:  list of L{Window}
        @return: Returns a list of windows
            handled by all processes in this snapshot.
        """
        window_list = list()
        for process in self.iter_processes():
            window_list.extend(process.get_windows())
        return window_list

    def get_pid_from_tid(self, dwThreadId):
        """
        Retrieves the global ID of the process that owns the thread.

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @rtype:  int
        @return: Process global ID.

        @raise KeyError: The thread does not exist.
        """
        try:
            try:
                hThread = win32.OpenThread(win32.THREAD_QUERY_LIMITED_INFORMATION, False, dwThreadId)
            except WindowsError:
                e = sys.exc_info()[1]
                if e.winerror != win32.ERROR_ACCESS_DENIED:
                    raise
                hThread = win32.OpenThread(win32.THREAD_QUERY_INFORMATION, False, dwThreadId)
            try:
                return win32.GetProcessIdOfThread(hThread)
            finally:
                hThread.close()
        except Exception:
            for aProcess in self.iter_processes():
                if aProcess.has_thread(dwThreadId):
                    return aProcess.get_pid()
        self.scan_processes_and_threads()
        for aProcess in self.iter_processes():
            if aProcess.has_thread(dwThreadId):
                return aProcess.get_pid()
        msg = 'Unknown thread ID %d' % dwThreadId
        raise KeyError(msg)

    @staticmethod
    def argv_to_cmdline(argv):
        """
        Convert a list of arguments to a single command line string.

        @type  argv: list( str )
        @param argv: List of argument strings.
            The first element is the program to execute.

        @rtype:  str
        @return: Command line string.
        """
        cmdline = list()
        for token in argv:
            if not token:
                token = '""'
            else:
                if '"' in token:
                    token = token.replace('"', '\\"')
                if ' ' in token or '\t' in token or '\n' in token or ('\r' in token):
                    token = '"%s"' % token
            cmdline.append(token)
        return ' '.join(cmdline)

    @staticmethod
    def cmdline_to_argv(lpCmdLine):
        """
        Convert a single command line string to a list of arguments.

        @type  lpCmdLine: str
        @param lpCmdLine: Command line string.
            The first token is the program to execute.

        @rtype:  list( str )
        @return: List of argument strings.
        """
        if not lpCmdLine:
            return []
        return win32.CommandLineToArgv(lpCmdLine)

    def start_process(self, lpCmdLine, **kwargs):
        """
        Starts a new process for instrumenting (or debugging).

        @type  lpCmdLine: str
        @param lpCmdLine: Command line to execute. Can't be an empty string.

        @type    bConsole: bool
        @keyword bConsole: True to inherit the console of the debugger.
            Defaults to C{False}.

        @type    bDebug: bool
        @keyword bDebug: C{True} to attach to the new process.
            To debug a process it's best to use the L{Debug} class instead.
            Defaults to C{False}.

        @type    bFollow: bool
        @keyword bFollow: C{True} to automatically attach to the child
            processes of the newly created process. Ignored unless C{bDebug} is
            C{True}. Defaults to C{False}.

        @type    bInheritHandles: bool
        @keyword bInheritHandles: C{True} if the new process should inherit
            it's parent process' handles. Defaults to C{False}.

        @type    bSuspended: bool
        @keyword bSuspended: C{True} to suspend the main thread before any code
            is executed in the debugee. Defaults to C{False}.

        @type    dwParentProcessId: int or None
        @keyword dwParentProcessId: C{None} if the debugger process should be
            the parent process (default), or a process ID to forcefully set as
            the debugee's parent (only available for Windows Vista and above).

        @type    iTrustLevel: int
        @keyword iTrustLevel: Trust level.
            Must be one of the following values:
             - 0: B{No trust}. May not access certain resources, such as
                  cryptographic keys and credentials. Only available since
                  Windows XP and 2003, desktop editions.
             - 1: B{Normal trust}. Run with the same privileges as a normal
                  user, that is, one that doesn't have the I{Administrator} or
                  I{Power User} user rights. Only available since Windows XP
                  and 2003, desktop editions.
             - 2: B{Full trust}. Run with the exact same privileges as the
                  current user. This is the default value.

        @type    bAllowElevation: bool
        @keyword bAllowElevation: C{True} to allow the child process to keep
            UAC elevation, if the debugger itself is running elevated. C{False}
            to ensure the child process doesn't run with elevation. Defaults to
            C{True}.

            This flag is only meaningful on Windows Vista and above, and if the
            debugger itself is running with elevation. It can be used to make
            sure the child processes don't run elevated as well.

            This flag DOES NOT force an elevation prompt when the debugger is
            not running with elevation.

            Note that running the debugger with elevation (or the Python
            interpreter at all for that matter) is not normally required.
            You should only need to if the target program requires elevation
            to work properly (for example if you try to debug an installer).

        @rtype:  L{Process}
        @return: Process object.
        """
        bConsole = kwargs.pop('bConsole', False)
        bDebug = kwargs.pop('bDebug', False)
        bFollow = kwargs.pop('bFollow', False)
        bSuspended = kwargs.pop('bSuspended', False)
        bInheritHandles = kwargs.pop('bInheritHandles', False)
        dwParentProcessId = kwargs.pop('dwParentProcessId', None)
        iTrustLevel = kwargs.pop('iTrustLevel', 2)
        bAllowElevation = kwargs.pop('bAllowElevation', True)
        if kwargs:
            raise TypeError('Unknown keyword arguments: %s' % compat.keys(kwargs))
        if not lpCmdLine:
            raise ValueError('Missing command line to execute!')
        if iTrustLevel is None:
            iTrustLevel = 2
        try:
            bAllowElevation = bAllowElevation or not self.is_admin()
        except AttributeError:
            bAllowElevation = True
            warnings.warn('UAC elevation is only available in Windows Vista and above', RuntimeWarning)
        dwCreationFlags = 0
        dwCreationFlags |= win32.CREATE_DEFAULT_ERROR_MODE
        dwCreationFlags |= win32.CREATE_BREAKAWAY_FROM_JOB
        if not bConsole:
            dwCreationFlags |= win32.DETACHED_PROCESS
        if bSuspended:
            dwCreationFlags |= win32.CREATE_SUSPENDED
        if bDebug:
            dwCreationFlags |= win32.DEBUG_PROCESS
            if not bFollow:
                dwCreationFlags |= win32.DEBUG_ONLY_THIS_PROCESS
        lpStartupInfo = None
        if dwParentProcessId is not None:
            myPID = win32.GetCurrentProcessId()
            if dwParentProcessId != myPID:
                if self.has_process(dwParentProcessId):
                    ParentProcess = self.get_process(dwParentProcessId)
                else:
                    ParentProcess = Process(dwParentProcessId)
                ParentProcessHandle = ParentProcess.get_handle(win32.PROCESS_CREATE_PROCESS)
                AttributeListData = ((win32.PROC_THREAD_ATTRIBUTE_PARENT_PROCESS, ParentProcessHandle._as_parameter_),)
                AttributeList = win32.ProcThreadAttributeList(AttributeListData)
                StartupInfoEx = win32.STARTUPINFOEX()
                StartupInfo = StartupInfoEx.StartupInfo
                StartupInfo.cb = win32.sizeof(win32.STARTUPINFOEX)
                StartupInfo.lpReserved = 0
                StartupInfo.lpDesktop = 0
                StartupInfo.lpTitle = 0
                StartupInfo.dwFlags = 0
                StartupInfo.cbReserved2 = 0
                StartupInfo.lpReserved2 = 0
                StartupInfoEx.lpAttributeList = AttributeList.value
                lpStartupInfo = StartupInfoEx
                dwCreationFlags |= win32.EXTENDED_STARTUPINFO_PRESENT
        pi = None
        try:
            if iTrustLevel >= 2 and bAllowElevation:
                pi = win32.CreateProcess(None, lpCmdLine, bInheritHandles=bInheritHandles, dwCreationFlags=dwCreationFlags, lpStartupInfo=lpStartupInfo)
            else:
                hToken = None
                try:
                    if not bAllowElevation:
                        if bFollow:
                            msg = "Child processes can't be autofollowed when dropping UAC elevation."
                            raise NotImplementedError(msg)
                        if bConsole:
                            msg = "Child processes can't inherit the debugger's console when dropping UAC elevation."
                            raise NotImplementedError(msg)
                        if bInheritHandles:
                            msg = "Child processes can't inherit the debugger's handles when dropping UAC elevation."
                            raise NotImplementedError(msg)
                        try:
                            hWnd = self.get_shell_window()
                        except WindowsError:
                            hWnd = self.get_desktop_window()
                        shell = hWnd.get_process()
                        try:
                            hShell = shell.get_handle(win32.PROCESS_QUERY_INFORMATION)
                            with win32.OpenProcessToken(hShell) as hShellToken:
                                hToken = win32.DuplicateTokenEx(hShellToken)
                        finally:
                            shell.close_handle()
                    if iTrustLevel < 2:
                        if iTrustLevel > 0:
                            dwLevelId = win32.SAFER_LEVELID_NORMALUSER
                        else:
                            dwLevelId = win32.SAFER_LEVELID_UNTRUSTED
                        with win32.SaferCreateLevel(dwLevelId=dwLevelId) as hSafer:
                            hSaferToken = win32.SaferComputeTokenFromLevel(hSafer, hToken)[0]
                            try:
                                if hToken is not None:
                                    hToken.close()
                            except:
                                hSaferToken.close()
                                raise
                            hToken = hSaferToken
                    if bAllowElevation:
                        pi = win32.CreateProcessAsUser(hToken=hToken, lpCommandLine=lpCmdLine, bInheritHandles=bInheritHandles, dwCreationFlags=dwCreationFlags, lpStartupInfo=lpStartupInfo)
                    else:
                        dwCreationFlags &= ~win32.DEBUG_PROCESS
                        dwCreationFlags &= ~win32.DEBUG_ONLY_THIS_PROCESS
                        dwCreationFlags &= ~win32.DETACHED_PROCESS
                        dwCreationFlags |= win32.CREATE_SUSPENDED
                        pi = win32.CreateProcessWithToken(hToken=hToken, dwLogonFlags=win32.LOGON_WITH_PROFILE, lpCommandLine=lpCmdLine, dwCreationFlags=dwCreationFlags, lpStartupInfo=lpStartupInfo)
                        if bDebug:
                            win32.DebugActiveProcess(pi.dwProcessId)
                        if not bSuspended:
                            win32.ResumeThread(pi.hThread)
                finally:
                    if hToken is not None:
                        hToken.close()
            aProcess = Process(pi.dwProcessId, pi.hProcess)
            aThread = Thread(pi.dwThreadId, pi.hThread)
            aProcess._add_thread(aThread)
            self._add_process(aProcess)
        except:
            if pi is not None:
                try:
                    win32.TerminateProcess(pi.hProcess)
                except WindowsError:
                    pass
                pi.hThread.close()
                pi.hProcess.close()
            raise
        return aProcess

    def get_explorer_pid(self):
        """
        Tries to find the process ID for "explorer.exe".

        @rtype:  int or None
        @return: Returns the process ID, or C{None} on error.
        """
        try:
            exp = win32.SHGetFolderPath(win32.CSIDL_WINDOWS)
        except Exception:
            exp = None
        if not exp:
            exp = os.getenv('SystemRoot')
        if exp:
            exp = os.path.join(exp, 'explorer.exe')
            exp_list = self.find_processes_by_filename(exp)
            if exp_list:
                return exp_list[0][0].get_pid()
        return None

    def scan(self):
        """
        Populates the snapshot with running processes and threads,
        and loaded modules.

        Tipically this is the first method called after instantiating a
        L{System} object, as it makes a best effort approach to gathering
        information on running processes.

        @rtype: bool
        @return: C{True} if the snapshot is complete, C{False} if the debugger
            doesn't have permission to scan some processes. In either case, the
            snapshot is complete for all processes the debugger has access to.
        """
        has_threads = True
        try:
            try:
                self.scan_processes_and_threads()
            except Exception:
                self.scan_processes_fast()
                for aProcess in self.__processDict.values():
                    if aProcess._get_thread_ids():
                        try:
                            aProcess.scan_threads()
                        except WindowsError:
                            has_threads = False
        finally:
            self.scan_processes()
        has_modules = self.scan_modules()
        has_full_names = self.scan_process_filenames()
        return has_threads and has_modules and has_full_names

    def scan_processes_and_threads(self):
        """
        Populates the snapshot with running processes and threads.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the Toolhelp API.

        @see: L{scan_modules}

        @raise WindowsError: An error occured while updating the snapshot.
            The snapshot was not modified.
        """
        our_pid = win32.GetCurrentProcessId()
        dead_pids = set(compat.iterkeys(self.__processDict))
        found_tids = set()
        if our_pid in dead_pids:
            dead_pids.remove(our_pid)
        dwFlags = win32.TH32CS_SNAPPROCESS | win32.TH32CS_SNAPTHREAD
        with win32.CreateToolhelp32Snapshot(dwFlags) as hSnapshot:
            pe = win32.Process32First(hSnapshot)
            while pe is not None:
                dwProcessId = pe.th32ProcessID
                if dwProcessId != our_pid:
                    if dwProcessId in dead_pids:
                        dead_pids.remove(dwProcessId)
                    if dwProcessId not in self.__processDict:
                        aProcess = Process(dwProcessId, fileName=pe.szExeFile)
                        self._add_process(aProcess)
                    elif pe.szExeFile:
                        aProcess = self.get_process(dwProcessId)
                        if not aProcess.fileName:
                            aProcess.fileName = pe.szExeFile
                pe = win32.Process32Next(hSnapshot)
            te = win32.Thread32First(hSnapshot)
            while te is not None:
                dwProcessId = te.th32OwnerProcessID
                if dwProcessId != our_pid:
                    if dwProcessId in dead_pids:
                        dead_pids.remove(dwProcessId)
                    if dwProcessId in self.__processDict:
                        aProcess = self.get_process(dwProcessId)
                    else:
                        aProcess = Process(dwProcessId)
                        self._add_process(aProcess)
                    dwThreadId = te.th32ThreadID
                    found_tids.add(dwThreadId)
                    if not aProcess._has_thread_id(dwThreadId):
                        aThread = Thread(dwThreadId, process=aProcess)
                        aProcess._add_thread(aThread)
                te = win32.Thread32Next(hSnapshot)
        for pid in dead_pids:
            self._del_process(pid)
        for aProcess in compat.itervalues(self.__processDict):
            dead_tids = set(aProcess._get_thread_ids())
            dead_tids.difference_update(found_tids)
            for tid in dead_tids:
                aProcess._del_thread(tid)

    def scan_modules(self):
        """
        Populates the snapshot with loaded modules.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the Toolhelp API.

        @see: L{scan_processes_and_threads}

        @rtype: bool
        @return: C{True} if the snapshot is complete, C{False} if the debugger
            doesn't have permission to scan some processes. In either case, the
            snapshot is complete for all processes the debugger has access to.
        """
        complete = True
        for aProcess in compat.itervalues(self.__processDict):
            try:
                aProcess.scan_modules()
            except WindowsError:
                complete = False
        return complete

    def scan_processes(self):
        """
        Populates the snapshot with running processes.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the Remote Desktop API instead of the Toolhelp
            API. It might give slightly different results, especially if the
            current process does not have full privileges.

        @note: This method will only retrieve process filenames. To get the
            process pathnames instead, B{after} this method call
            L{scan_process_filenames}.

        @raise WindowsError: An error occured while updating the snapshot.
            The snapshot was not modified.
        """
        our_pid = win32.GetCurrentProcessId()
        dead_pids = set(compat.iterkeys(self.__processDict))
        if our_pid in dead_pids:
            dead_pids.remove(our_pid)
        pProcessInfo = None
        try:
            pProcessInfo, dwCount = win32.WTSEnumerateProcesses(win32.WTS_CURRENT_SERVER_HANDLE)
            for index in compat.xrange(dwCount):
                sProcessInfo = pProcessInfo[index]
                pid = sProcessInfo.ProcessId
                if pid == our_pid:
                    continue
                if pid in dead_pids:
                    dead_pids.remove(pid)
                fileName = sProcessInfo.pProcessName
                if pid not in self.__processDict:
                    aProcess = Process(pid, fileName=fileName)
                    self._add_process(aProcess)
                elif fileName:
                    aProcess = self.__processDict.get(pid)
                    if not aProcess.fileName:
                        aProcess.fileName = fileName
        finally:
            if pProcessInfo is not None:
                try:
                    win32.WTSFreeMemory(pProcessInfo)
                except WindowsError:
                    pass
        for pid in dead_pids:
            self._del_process(pid)

    def scan_processes_fast(self):
        """
        Populates the snapshot with running processes.
        Only the PID is retrieved for each process.

        Dead processes are removed.
        Threads and modules of living processes are ignored.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the PSAPI. It may be faster for scanning,
            but some information may be missing, outdated or slower to obtain.
            This could be a good tradeoff under some circumstances.
        """
        new_pids = set(win32.EnumProcesses())
        old_pids = set(compat.iterkeys(self.__processDict))
        our_pid = win32.GetCurrentProcessId()
        if our_pid in new_pids:
            new_pids.remove(our_pid)
        if our_pid in old_pids:
            old_pids.remove(our_pid)
        for pid in new_pids.difference(old_pids):
            self._add_process(Process(pid))
        for pid in old_pids.difference(new_pids):
            self._del_process(pid)

    def scan_process_filenames(self):
        """
        Update the filename for each process in the snapshot when possible.

        @note: Tipically you don't need to call this method. It's called
            automatically by L{scan} to get the full pathname for each process
            when possible, since some scan methods only get filenames without
            the path component.

            If unsure, use L{scan} instead.

        @see: L{scan}, L{Process.get_filename}

        @rtype: bool
        @return: C{True} if all the pathnames were retrieved, C{False} if the
            debugger doesn't have permission to scan some processes. In either
            case, all processes the debugger has access to have a full pathname
            instead of just a filename.
        """
        complete = True
        for aProcess in self.__processDict.values():
            try:
                new_name = None
                old_name = aProcess.fileName
                try:
                    aProcess.fileName = None
                    new_name = aProcess.get_filename()
                finally:
                    if not new_name:
                        aProcess.fileName = old_name
                        complete = False
            except Exception:
                complete = False
        return complete

    def clear_dead_processes(self):
        """
        Removes Process objects from the snapshot
        referring to processes no longer running.
        """
        for pid in self.get_process_ids():
            aProcess = self.get_process(pid)
            if not aProcess.is_alive():
                self._del_process(aProcess)

    def clear_unattached_processes(self):
        """
        Removes Process objects from the snapshot
        referring to processes not being debugged.
        """
        for pid in self.get_process_ids():
            aProcess = self.get_process(pid)
            if not aProcess.is_being_debugged():
                self._del_process(aProcess)

    def close_process_handles(self):
        """
        Closes all open handles to processes in this snapshot.
        """
        for pid in self.get_process_ids():
            aProcess = self.get_process(pid)
            try:
                aProcess.close_handle()
            except Exception:
                e = sys.exc_info()[1]
                try:
                    msg = 'Cannot close process handle %s, reason: %s'
                    msg %= (aProcess.hProcess.value, str(e))
                    warnings.warn(msg)
                except Exception:
                    pass

    def close_process_and_thread_handles(self):
        """
        Closes all open handles to processes and threads in this snapshot.
        """
        for aProcess in self.iter_processes():
            aProcess.close_thread_handles()
            try:
                aProcess.close_handle()
            except Exception:
                e = sys.exc_info()[1]
                try:
                    msg = 'Cannot close process handle %s, reason: %s'
                    msg %= (aProcess.hProcess.value, str(e))
                    warnings.warn(msg)
                except Exception:
                    pass

    def clear_processes(self):
        """
        Removes all L{Process}, L{Thread} and L{Module} objects in this snapshot.
        """
        for aProcess in self.iter_processes():
            aProcess.clear()
        self.__processDict = dict()

    def clear(self):
        """
        Clears this snapshot.

        @see: L{clear_processes}
        """
        self.clear_processes()

    def has_thread(self, dwThreadId):
        dwProcessId = self.get_pid_from_tid(dwThreadId)
        if dwProcessId is None:
            return False
        return self.has_process(dwProcessId)

    def get_thread(self, dwThreadId):
        dwProcessId = self.get_pid_from_tid(dwThreadId)
        if dwProcessId is None:
            msg = 'Unknown thread ID %d' % dwThreadId
            raise KeyError(msg)
        return self.get_process(dwProcessId).get_thread(dwThreadId)

    def get_thread_ids(self):
        ids = list()
        for aProcess in self.iter_processes():
            ids += aProcess.get_thread_ids()
        return ids

    def get_thread_count(self):
        count = 0
        for aProcess in self.iter_processes():
            count += aProcess.get_thread_count()
        return count
    has_thread.__doc__ = _ThreadContainer.has_thread.__doc__
    get_thread.__doc__ = _ThreadContainer.get_thread.__doc__
    get_thread_ids.__doc__ = _ThreadContainer.get_thread_ids.__doc__
    get_thread_count.__doc__ = _ThreadContainer.get_thread_count.__doc__

    def get_module_count(self):
        count = 0
        for aProcess in self.iter_processes():
            count += aProcess.get_module_count()
        return count
    get_module_count.__doc__ = _ModuleContainer.get_module_count.__doc__

    def find_modules_by_base(self, lpBaseOfDll):
        """
        @rtype:  list( L{Module}... )
        @return: List of Module objects with the given base address.
        """
        found = list()
        for aProcess in self.iter_processes():
            if aProcess.has_module(lpBaseOfDll):
                aModule = aProcess.get_module(lpBaseOfDll)
                found.append((aProcess, aModule))
        return found

    def find_modules_by_name(self, fileName):
        """
        @rtype:  list( L{Module}... )
        @return: List of Module objects found.
        """
        found = list()
        for aProcess in self.iter_processes():
            aModule = aProcess.get_module_by_name(fileName)
            if aModule is not None:
                found.append((aProcess, aModule))
        return found

    def find_modules_by_address(self, address):
        """
        @rtype:  list( L{Module}... )
        @return: List of Module objects that best match the given address.
        """
        found = list()
        for aProcess in self.iter_processes():
            aModule = aProcess.get_module_at_address(address)
            if aModule is not None:
                found.append((aProcess, aModule))
        return found

    def __find_processes_by_filename(self, filename):
        """
        Internally used by L{find_processes_by_filename}.
        """
        found = list()
        filename = filename.lower()
        if PathOperations.path_is_absolute(filename):
            for aProcess in self.iter_processes():
                imagename = aProcess.get_filename()
                if imagename and imagename.lower() == filename:
                    found.append((aProcess, imagename))
        else:
            for aProcess in self.iter_processes():
                imagename = aProcess.get_filename()
                if imagename:
                    imagename = PathOperations.pathname_to_filename(imagename)
                    if imagename.lower() == filename:
                        found.append((aProcess, imagename))
        return found

    def find_processes_by_filename(self, fileName):
        """
        @type  fileName: str
        @param fileName: Filename to search for.
            If it's a full pathname, the match must be exact.
            If it's a base filename only, the file part is matched,
            regardless of the directory where it's located.

        @note: If the process is not found and the file extension is not
            given, this method will search again assuming a default
            extension (.exe).

        @rtype:  list of tuple( L{Process}, str )
        @return: List of processes matching the given main module filename.
            Each tuple contains a Process object and it's filename.
        """
        found = self.__find_processes_by_filename(fileName)
        if not found:
            fn, ext = PathOperations.split_extension(fileName)
            if not ext:
                fileName = '%s.exe' % fn
                found = self.__find_processes_by_filename(fileName)
        return found

    def _add_process(self, aProcess):
        """
        Private method to add a process object to the snapshot.

        @type  aProcess: L{Process}
        @param aProcess: Process object.
        """
        dwProcessId = aProcess.dwProcessId
        self.__processDict[dwProcessId] = aProcess

    def _del_process(self, dwProcessId):
        """
        Private method to remove a process object from the snapshot.

        @type  dwProcessId: int
        @param dwProcessId: Global process ID.
        """
        try:
            aProcess = self.__processDict[dwProcessId]
            del self.__processDict[dwProcessId]
        except KeyError:
            aProcess = None
            msg = 'Unknown process ID %d' % dwProcessId
            warnings.warn(msg, RuntimeWarning)
        if aProcess:
            aProcess.clear()

    def _notify_create_process(self, event):
        """
        Notify the creation of a new process.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{CreateProcessEvent}
        @param event: Create process event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        dwProcessId = event.get_pid()
        dwThreadId = event.get_tid()
        hProcess = event.get_process_handle()
        if dwProcessId not in self.__processDict:
            aProcess = Process(dwProcessId, hProcess)
            self._add_process(aProcess)
            aProcess.fileName = event.get_filename()
        else:
            aProcess = self.get_process(dwProcessId)
            if not aProcess.fileName:
                fileName = event.get_filename()
                if fileName:
                    aProcess.fileName = fileName
        return aProcess._notify_create_process(event)

    def _notify_exit_process(self, event):
        """
        Notify the termination of a process.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{ExitProcessEvent}
        @param event: Exit process event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        dwProcessId = event.get_pid()
        if dwProcessId in self.__processDict:
            self._del_process(dwProcessId)
        return True