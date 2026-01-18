from abc import ABC
from abc import abstractmethod
import errno
import os
class _WindowsInterProcessReaderWriterLockMechanism(_InterProcessReaderWriterLockMechanism):
    """Interprocess readers writer lock implementation that works on windows
    systems."""

    @staticmethod
    def trylock(lockfile, exclusive):
        if exclusive:
            flags = win32con.LOCKFILE_FAIL_IMMEDIATELY | win32con.LOCKFILE_EXCLUSIVE_LOCK
        else:
            flags = win32con.LOCKFILE_FAIL_IMMEDIATELY
        handle = msvcrt.get_osfhandle(lockfile.fileno())
        ok = win32file.LockFileEx(handle, flags, 0, 1, 0, win32file.pointer(pywintypes.OVERLAPPED()))
        if ok:
            return True
        else:
            last_error = win32file.GetLastError()
            if last_error == win32file.ERROR_LOCK_VIOLATION:
                return False
            else:
                raise OSError(last_error)

    @staticmethod
    def unlock(lockfile):
        handle = msvcrt.get_osfhandle(lockfile.fileno())
        ok = win32file.UnlockFileEx(handle, 0, 1, 0, win32file.pointer(pywintypes.OVERLAPPED()))
        if not ok:
            raise OSError(win32file.GetLastError())

    @staticmethod
    def get_handle(path):
        return open(path, 'a+')

    @staticmethod
    def close_handle(lockfile):
        lockfile.close()