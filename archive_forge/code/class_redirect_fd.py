import io
import logging
import os
import sys
import threading
import time
from io import StringIO
class redirect_fd(object):
    """Redirect a file descriptor to a new file or file descriptor.

    This context manager will redirect the specified file descriptor to
    a specified new output target (either file name or file descriptor).
    For the special case of file descriptors 1 (stdout) and 2 (stderr),
    we will also make sure that the Python `sys.stdout` or `sys.stderr`
    remain usable: in the case of synchronize=True, the `sys.stdout` /
    `sys.stderr` file handles point to the new file descriptor.  When
    synchronize=False, we preserve the behavior of the Python file
    object (retargeting it to the original file descriptor if necessary).

    Parameters
    ----------
    fd: int
        The file descriptor to redirect

    output: int or str
        The new output target for `fd`: either another valid file
        descriptor (int) or a string with the file to open.

    synchronize: bool
        If True, and `fd` is 1 or 2, then update `sys.stdout` or
        `sys.stderr` to also point to the new file descriptor
    """

    def __init__(self, fd=1, output=None, synchronize=True):
        if output is None:
            output = os.devnull
        self.fd = fd
        self.std = {1: 'stdout', 2: 'stderr'}.get(self.fd, None)
        self.target = output
        self.target_file = None
        self.synchronize = synchronize
        self.original_file = None
        self.original_fd = None

    def __enter__(self):
        if self.std:
            getattr(sys, self.std).flush()
            self.original_file = getattr(sys, self.std)
        self.original_fd = os.dup(self.fd)
        if isinstance(self.target, int):
            out_fd = self.target
        else:
            out_fd = os.open(self.target, os.O_WRONLY)
        os.dup2(out_fd, self.fd, inheritable=bool(self.std))
        if out_fd is not self.target:
            os.close(out_fd)
        if self.std:
            if self.synchronize:
                fd = self.fd
            else:
                try:
                    old_std_fd = getattr(sys, self.std).fileno()
                    fd = self.original_fd if old_std_fd == self.fd else None
                except (io.UnsupportedOperation, AttributeError):
                    fd = None
            if fd is not None:
                self.target_file = os.fdopen(fd, 'w', closefd=False)
                setattr(sys, self.std, self.target_file)
        return self

    def __exit__(self, t, v, traceback):
        if self.target_file is not None:
            self.target_file.flush()
            self.target_file.close()
            self.target_file = None
            setattr(sys, self.std, self.original_file)
        os.dup2(self.original_fd, self.fd, inheritable=bool(self.std))
        os.close(self.original_fd)