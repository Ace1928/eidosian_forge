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
def __read_pidfile(pidfile, delete_if_stale):
    if _pidfile_dev is not None:
        try:
            s = os.stat(pidfile)
            if s.st_ino == _pidfile_ino and s.st_dev == _pidfile_dev:
                return os.getpid()
        except OSError:
            pass
    try:
        file_handle = open(pidfile, 'r+')
    except IOError as e:
        if e.errno == errno.ENOENT and delete_if_stale:
            return 0
        vlog.warn('%s: open: %s' % (pidfile, e.strerror))
        return -e.errno
    try:
        fcntl.lockf(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not delete_if_stale:
            file_handle.close()
            vlog.warn('%s: pid file is stale' % pidfile)
            return -errno.ESRCH
        try:
            raced = False
            s = os.stat(pidfile)
            s2 = os.fstat(file_handle.fileno())
            if s.st_ino != s2.st_ino or s.st_dev != s2.st_dev:
                raced = True
        except IOError:
            raced = True
        if raced:
            vlog.warn('%s: lost race to delete pidfile' % pidfile)
            return -errno.EALREADY
        try:
            os.unlink(pidfile)
        except IOError as e:
            vlog.warn('%s: failed to delete stale pidfile (%s)' % (pidfile, e.strerror))
            return -e.errno
        else:
            vlog.dbg('%s: deleted stale pidfile' % pidfile)
            file_handle.close()
            return 0
    except IOError as e:
        if e.errno not in [errno.EACCES, errno.EAGAIN]:
            vlog.warn('%s: fcntl: %s' % (pidfile, e.strerror))
            return -e.errno
    try:
        try:
            error = int(file_handle.readline())
        except IOError as e:
            vlog.warn('%s: read: %s' % (pidfile, e.strerror))
            error = -e.errno
        except ValueError:
            vlog.warn('%s does not contain a pid' % pidfile)
            error = -errno.EINVAL
        return error
    finally:
        try:
            file_handle.close()
        except IOError:
            pass