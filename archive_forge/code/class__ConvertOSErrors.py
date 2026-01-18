from __future__ import print_function, unicode_literals
import sys
import typing
import errno
import platform
from contextlib import contextmanager
from six import reraise
from . import errors
class _ConvertOSErrors(object):
    """Context manager to convert OSErrors in to FS Errors."""
    FILE_ERRORS = {64: errors.RemoteConnectionError, errno.EACCES: errors.PermissionDenied, errno.ENOENT: errors.ResourceNotFound, errno.EFAULT: errors.ResourceNotFound, errno.ESRCH: errors.ResourceNotFound, errno.ENOTEMPTY: errors.DirectoryNotEmpty, errno.EEXIST: errors.FileExists, 183: errors.DirectoryExists, errno.ENOTDIR: errors.ResourceNotFound, errno.EISDIR: errors.FileExpected, errno.EINVAL: errors.FileExpected, errno.ENOSPC: errors.InsufficientStorage, errno.EPERM: errors.PermissionDenied, errno.ENETDOWN: errors.RemoteConnectionError, errno.ECONNRESET: errors.RemoteConnectionError, errno.ENAMETOOLONG: errors.PathError, errno.EOPNOTSUPP: errors.Unsupported, errno.ENOSYS: errors.Unsupported}
    DIR_ERRORS = FILE_ERRORS.copy()
    DIR_ERRORS[errno.ENOTDIR] = errors.DirectoryExpected
    DIR_ERRORS[errno.EEXIST] = errors.DirectoryExists
    DIR_ERRORS[errno.EINVAL] = errors.DirectoryExpected
    if _WINDOWS_PLATFORM:
        DIR_ERRORS[13] = errors.DirectoryExpected
        DIR_ERRORS[267] = errors.DirectoryExpected
        FILE_ERRORS[13] = errors.FileExpected

    def __init__(self, opname, path, directory=False):
        self._opname = opname
        self._path = path
        self._directory = directory

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os_errors = self.DIR_ERRORS if self._directory else self.FILE_ERRORS
        if exc_type and isinstance(exc_value, EnvironmentError):
            _errno = exc_value.errno
            fserror = os_errors.get(_errno, errors.OperationFailed)
            if _errno == errno.EACCES and sys.platform == 'win32':
                if getattr(exc_value, 'args', None) == 32:
                    fserror = errors.ResourceLocked
            reraise(fserror, fserror(self._path, exc=exc_value), traceback)