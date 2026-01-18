import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
class Win32ShellCommandController(object):
    """Runs a shell command in a 'with' context.

    This implementation is Win32-specific.

    Example:
        # Runs the command interactively with default console stdin/stdout
        with ShellCommandController('python -i') as scc:
            scc.run()

        # Runs the command using the provided functions for stdin/stdout
        def my_stdout_func(s):
            # print or save the string 's'
            write_to_stdout(s)
        def my_stdin_func():
            # If input is available, return it as a string.
            if input_available():
                return get_input()
            # If no input available, return None after a short delay to
            # keep from blocking.
            else:
                time.sleep(0.01)
                return None
      
        with ShellCommandController('python -i') as scc:
            scc.run(my_stdout_func, my_stdin_func)
    """

    def __init__(self, cmd, mergeout=True):
        """Initializes the shell command controller.

        The cmd is the program to execute, and mergeout is
        whether to blend stdout and stderr into one output
        in stdout. Merging them together in this fashion more
        reliably keeps stdout and stderr in the correct order
        especially for interactive shell usage.
        """
        self.cmd = cmd
        self.mergeout = mergeout

    def __enter__(self):
        cmd = self.cmd
        mergeout = self.mergeout
        self.hstdout, self.hstdin, self.hstderr = (None, None, None)
        self.piProcInfo = None
        try:
            p_hstdout, c_hstdout, p_hstderr, c_hstderr, p_hstdin, c_hstdin = [None] * 6
            saAttr = SECURITY_ATTRIBUTES()
            saAttr.nLength = ctypes.sizeof(saAttr)
            saAttr.bInheritHandle = True
            saAttr.lpSecurityDescriptor = None

            def create_pipe(uninherit):
                """Creates a Windows pipe, which consists of two handles.

                The 'uninherit' parameter controls which handle is not
                inherited by the child process.
                """
                handles = (HANDLE(), HANDLE())
                if not CreatePipe(ctypes.byref(handles[0]), ctypes.byref(handles[1]), ctypes.byref(saAttr), 0):
                    raise ctypes.WinError()
                if not SetHandleInformation(handles[uninherit], HANDLE_FLAG_INHERIT, 0):
                    raise ctypes.WinError()
                return (handles[0].value, handles[1].value)
            p_hstdout, c_hstdout = create_pipe(uninherit=0)
            if mergeout:
                c_hstderr = HANDLE()
                if not DuplicateHandle(GetCurrentProcess(), c_hstdout, GetCurrentProcess(), ctypes.byref(c_hstderr), 0, True, DUPLICATE_SAME_ACCESS):
                    raise ctypes.WinError()
            else:
                p_hstderr, c_hstderr = create_pipe(uninherit=0)
            c_hstdin, p_hstdin = create_pipe(uninherit=1)
            piProcInfo = PROCESS_INFORMATION()
            siStartInfo = STARTUPINFO()
            siStartInfo.cb = ctypes.sizeof(siStartInfo)
            siStartInfo.hStdInput = c_hstdin
            siStartInfo.hStdOutput = c_hstdout
            siStartInfo.hStdError = c_hstderr
            siStartInfo.dwFlags = STARTF_USESTDHANDLES
            dwCreationFlags = CREATE_SUSPENDED | CREATE_NO_WINDOW
            if not CreateProcess(None, u'cmd.exe /c ' + cmd, None, None, True, dwCreationFlags, None, None, ctypes.byref(siStartInfo), ctypes.byref(piProcInfo)):
                raise ctypes.WinError()
            CloseHandle(c_hstdin)
            c_hstdin = None
            CloseHandle(c_hstdout)
            c_hstdout = None
            if c_hstderr is not None:
                CloseHandle(c_hstderr)
                c_hstderr = None
            self.hstdin = p_hstdin
            p_hstdin = None
            self.hstdout = p_hstdout
            p_hstdout = None
            if not mergeout:
                self.hstderr = p_hstderr
                p_hstderr = None
            self.piProcInfo = piProcInfo
        finally:
            if p_hstdin:
                CloseHandle(p_hstdin)
            if c_hstdin:
                CloseHandle(c_hstdin)
            if p_hstdout:
                CloseHandle(p_hstdout)
            if c_hstdout:
                CloseHandle(c_hstdout)
            if p_hstderr:
                CloseHandle(p_hstderr)
            if c_hstderr:
                CloseHandle(c_hstderr)
        return self

    def _stdin_thread(self, handle, hprocess, func, stdout_func):
        exitCode = DWORD()
        bytesWritten = DWORD(0)
        while True:
            data = func()
            if data is None:
                if not GetExitCodeProcess(hprocess, ctypes.byref(exitCode)):
                    raise ctypes.WinError()
                if exitCode.value != STILL_ACTIVE:
                    return
                if not WriteFile(handle, '', 0, ctypes.byref(bytesWritten), None):
                    raise ctypes.WinError()
                continue
            if isinstance(data, unicode):
                data = data.encode('utf_8')
            if not isinstance(data, str):
                raise RuntimeError('internal stdin function string error')
            if len(data) == 0:
                return
            stdout_func(data)
            while len(data) != 0:
                if not WriteFile(handle, data, len(data), ctypes.byref(bytesWritten), None):
                    if GetLastError() == ERROR_NO_DATA:
                        return
                    raise ctypes.WinError()
                data = data[bytesWritten.value:]

    def _stdout_thread(self, handle, func):
        data = ctypes.create_string_buffer(4096)
        while True:
            bytesRead = DWORD(0)
            if not ReadFile(handle, data, 4096, ctypes.byref(bytesRead), None):
                le = GetLastError()
                if le == ERROR_BROKEN_PIPE:
                    return
                else:
                    raise ctypes.WinError()
            s = data.value[0:bytesRead.value]
            func(s.decode('utf_8', 'replace'))

    def run(self, stdout_func=None, stdin_func=None, stderr_func=None):
        """Runs the process, using the provided functions for I/O.

        The function stdin_func should return strings whenever a
        character or characters become available.
        The functions stdout_func and stderr_func are called whenever
        something is printed to stdout or stderr, respectively.
        These functions are called from different threads (but not
        concurrently, because of the GIL).
        """
        if stdout_func is None and stdin_func is None and (stderr_func is None):
            return self._run_stdio()
        if stderr_func is not None and self.mergeout:
            raise RuntimeError('Shell command was initiated with merged stdin/stdout, but a separate stderr_func was provided to the run() method')
        stdin_thread = None
        threads = []
        if stdin_func:
            stdin_thread = threading.Thread(target=self._stdin_thread, args=(self.hstdin, self.piProcInfo.hProcess, stdin_func, stdout_func))
        threads.append(threading.Thread(target=self._stdout_thread, args=(self.hstdout, stdout_func)))
        if not self.mergeout:
            if stderr_func is None:
                stderr_func = stdout_func
            threads.append(threading.Thread(target=self._stdout_thread, args=(self.hstderr, stderr_func)))
        if ResumeThread(self.piProcInfo.hThread) == 4294967295:
            raise ctypes.WinError()
        if stdin_thread is not None:
            stdin_thread.start()
        for thread in threads:
            thread.start()
        if WaitForSingleObject(self.piProcInfo.hProcess, INFINITE) == WAIT_FAILED:
            raise ctypes.WinError()
        for thread in threads:
            thread.join()
        if stdin_thread is not None:
            stdin_thread.join()

    def _stdin_raw_nonblock(self):
        """Use the raw Win32 handle of sys.stdin to do non-blocking reads"""
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        result = WaitForSingleObject(handle, 100)
        if result == WAIT_FAILED:
            raise ctypes.WinError()
        elif result == WAIT_TIMEOUT:
            print('.', end='')
            return None
        else:
            data = ctypes.create_string_buffer(256)
            bytesRead = DWORD(0)
            print('?', end='')
            if not ReadFile(handle, data, 256, ctypes.byref(bytesRead), None):
                raise ctypes.WinError()
            FlushConsoleInputBuffer(handle)
            data = data.value
            data = data.replace('\r\n', '\n')
            data = data.replace('\r', '\n')
            print(repr(data) + ' ', end='')
            return data

    def _stdin_raw_block(self):
        """Use a blocking stdin read"""
        try:
            data = sys.stdin.read(1)
            data = data.replace('\r', '\n')
            return data
        except WindowsError as we:
            if we.winerror == ERROR_NO_DATA:
                return None
            else:
                raise we

    def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

    def _stderr_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stderr)
        sys.stderr.flush()

    def _run_stdio(self):
        """Runs the process using the system standard I/O.

        IMPORTANT: stdin needs to be asynchronous, so the Python
                   sys.stdin object is not used. Instead,
                   msvcrt.kbhit/getwch are used asynchronously.
        """
        if self.mergeout:
            return self.run(stdout_func=self._stdout_raw, stdin_func=self._stdin_raw_block)
        else:
            return self.run(stdout_func=self._stdout_raw, stdin_func=self._stdin_raw_block, stderr_func=self._stderr_raw)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hstdin:
            CloseHandle(self.hstdin)
            self.hstdin = None
        if self.hstdout:
            CloseHandle(self.hstdout)
            self.hstdout = None
        if self.hstderr:
            CloseHandle(self.hstderr)
            self.hstderr = None
        if self.piProcInfo is not None:
            CloseHandle(self.piProcInfo.hProcess)
            CloseHandle(self.piProcInfo.hThread)
            self.piProcInfo = None