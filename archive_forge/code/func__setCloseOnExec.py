import errno
import os
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
def _setCloseOnExec(fd):
    """
        Make a file descriptor close-on-exec.
        """
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    flags = flags | fcntl.FD_CLOEXEC
    fcntl.fcntl(fd, fcntl.F_SETFD, flags)