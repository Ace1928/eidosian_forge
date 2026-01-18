from winappdbg import win32
def __get_pid_and_tid(self):
    """Internally used by get_pid() and get_tid()."""
    self.dwThreadId, self.dwProcessId = win32.GetWindowThreadProcessId(self.get_handle())