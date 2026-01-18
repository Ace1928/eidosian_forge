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
def _fork_and_wait_for_startup_windows():
    global _detached
    if _detached:
        ovs.timeval.postfork()
        return 0
    ' close the log file, on Windows cannot be moved while the parent has\n    a reference on it.'
    vlog.close_log_file()
    try:
        rfd, wfd = winutils.windows_create_pipe()
    except pywintypes.error as e:
        sys.stderr.write('pipe failed to create: %s\n' % e.strerror)
        sys.exit(1)
    try:
        creationFlags = win32process.DETACHED_PROCESS
        args = '%s %s --pipe-handle=%ld' % (sys.executable, ' '.join(sys.argv), int(wfd))
        proc = subprocess.Popen(args=args, close_fds=False, shell=False, creationflags=creationFlags, stdout=sys.stdout, stderr=sys.stderr)
        pid = proc.pid
    except OSError as e:
        sys.stderr.write('CreateProcess failed (%s)\n' % os.strerror(e.errno))
        sys.exit(1)
    winutils.win32file.CloseHandle(wfd)
    ovs.fatal_signal.fork()
    error, s = winutils.windows_read_pipe(rfd, 1)
    if error:
        s = ''
    if len(s) != 1:
        retval = proc.wait()
        if retval == 0:
            sys.stderr.write('fork child failed to signal startup\n')
        else:
            sys.exit(retval)
    winutils.win32file.CloseHandle(rfd)
    return pid