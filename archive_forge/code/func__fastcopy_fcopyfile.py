import os
import sys
import stat
import fnmatch
import collections
import errno
def _fastcopy_fcopyfile(fsrc, fdst, flags):
    """Copy a regular file content or metadata by using high-performance
    fcopyfile(3) syscall (macOS).
    """
    try:
        infd = fsrc.fileno()
        outfd = fdst.fileno()
    except Exception as err:
        raise _GiveupOnFastCopy(err)
    try:
        posix._fcopyfile(infd, outfd, flags)
    except OSError as err:
        err.filename = fsrc.name
        err.filename2 = fdst.name
        if err.errno in {errno.EINVAL, errno.ENOTSUP}:
            raise _GiveupOnFastCopy(err)
        else:
            raise err from None