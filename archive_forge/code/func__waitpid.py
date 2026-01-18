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
def _waitpid(pid, options):
    while True:
        try:
            return os.waitpid(pid, options)
        except OSError as e:
            if e.errno == errno.EINTR:
                pass
            return (-e.errno, 0)