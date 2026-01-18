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
def _make_pidfile():
    """If a pidfile has been configured, creates it and stores the running
    process's pid in it.  Ensures that the pidfile will be deleted when the
    process exits."""
    pid = os.getpid()
    if sys.platform != 'win32':
        tmpfile = '%s.tmp%d' % (_pidfile, pid)
        ovs.fatal_signal.add_file_to_unlink(tmpfile)
    else:
        tmpfile = '%s' % _pidfile
    try:
        global file_handle
        file_handle = open(tmpfile, 'w')
    except IOError as e:
        _fatal('%s: create failed (%s)' % (tmpfile, e.strerror))
    try:
        s = os.fstat(file_handle.fileno())
    except IOError as e:
        _fatal('%s: fstat failed (%s)' % (tmpfile, e.strerror))
    try:
        file_handle.write('%s\n' % pid)
        file_handle.flush()
    except OSError as e:
        _fatal('%s: write failed: %s' % (tmpfile, e.strerror))
    try:
        if sys.platform != 'win32':
            fcntl.lockf(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            fcntl.lockf(file_handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except IOError as e:
        _fatal('%s: fcntl failed: %s' % (tmpfile, e.strerror))
    if sys.platform == 'win32':
        ovs.fatal_signal.add_file_to_close_and_unlink(_pidfile, file_handle)
    else:
        if _overwrite_pidfile:
            try:
                os.rename(tmpfile, _pidfile)
            except OSError as e:
                _fatal('failed to rename "%s" to "%s" (%s)' % (tmpfile, _pidfile, e.strerror))
        else:
            while True:
                try:
                    os.link(tmpfile, _pidfile)
                    error = 0
                except OSError as e:
                    error = e.errno
                if error == errno.EEXIST:
                    _check_already_running()
                elif error != errno.EINTR:
                    break
            if error:
                _fatal('failed to link "%s" as "%s" (%s)' % (tmpfile, _pidfile, os.strerror(error)))
        ovs.fatal_signal.add_file_to_unlink(_pidfile)
        if not _overwrite_pidfile:
            error = ovs.fatal_signal.unlink_file_now(tmpfile)
            if error:
                _fatal('%s: unlink failed (%s)' % (tmpfile, os.strerror(error)))
    global _pidfile_dev
    global _pidfile_ino
    _pidfile_dev = s.st_dev
    _pidfile_ino = s.st_ino