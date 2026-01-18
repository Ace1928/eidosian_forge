import logging
import threading
class ReadWriteMutex(object):
    """A mutex which allows multiple readers, single writer.

    :class:`.ReadWriteMutex` uses a Python ``threading.Condition``
    to provide this functionality across threads within a process.

    The Beaker package also contained a file-lock based version
    of this concept, so that readers/writers could be synchronized
    across processes with a common filesystem.  A future Dogpile
    release may include this additional class at some point.

    """

    def __init__(self):
        self.async_ = 0
        self.current_sync_operation = None
        self.condition = threading.Condition(threading.Lock())

    def acquire_read_lock(self, wait=True):
        """Acquire the 'read' lock."""
        self.condition.acquire()
        try:
            if wait:
                while self.current_sync_operation is not None:
                    self.condition.wait()
            elif self.current_sync_operation is not None:
                return False
            self.async_ += 1
            log.debug('%s acquired read lock', self)
        finally:
            self.condition.release()
        if not wait:
            return True

    def release_read_lock(self):
        """Release the 'read' lock."""
        self.condition.acquire()
        try:
            self.async_ -= 1
            if self.async_ == 0:
                if self.current_sync_operation is not None:
                    self.condition.notify_all()
            elif self.async_ < 0:
                raise LockError('Synchronizer error - too many release_read_locks called')
            log.debug('%s released read lock', self)
        finally:
            self.condition.release()

    def acquire_write_lock(self, wait=True):
        """Acquire the 'write' lock."""
        self.condition.acquire()
        try:
            if wait:
                while self.current_sync_operation is not None:
                    self.condition.wait()
            elif self.current_sync_operation is not None:
                return False
            self.current_sync_operation = threading.current_thread()
            if self.async_ > 0:
                if wait:
                    self.condition.wait()
                else:
                    self.current_sync_operation = None
                    return False
            log.debug('%s acquired write lock', self)
        finally:
            self.condition.release()
        if not wait:
            return True

    def release_write_lock(self):
        """Release the 'write' lock."""
        self.condition.acquire()
        try:
            if self.current_sync_operation is not threading.current_thread():
                raise LockError("Synchronizer error - current thread doesn't have the write lock")
            self.current_sync_operation = None
            self.condition.notify_all()
            log.debug('%s released write lock', self)
        finally:
            self.condition.release()