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
def _fork_notify_startup_windows(fd):
    if fd is not None:
        try:
            winutils.win32file.WriteFile(fd, b'0', None)
        except winutils.pywintypes.error as e:
            sys.stderr.write('could not write to pipe: %s\n' % os.strerror(e.winerror))
            sys.exit(1)