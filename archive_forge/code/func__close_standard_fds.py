import errno
import os
import signal
import sys
import time
import ovs.dirs
import ovs.fatal_signal
import ovs.process
import ovs.socket_util
import ovs.timeval
import ovs.util
import ovs.vlog
def _close_standard_fds():
    """Close stdin, stdout, stderr.  If we're started from e.g. an SSH session,
    then this keeps us from holding that session open artificially."""
    null_fd = ovs.socket_util.get_null_fd()
    if null_fd >= 0:
        os.dup2(null_fd, 0)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)