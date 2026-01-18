from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
class AbstractFileLock:
    """Coordinate read/write access to a file.

    typically is a file-based lock but doesn't necessarily have to be.

    The default implementation here is :class:`.FileLock`.

    Implementations should provide the following methods::

        * __init__()
        * acquire_read_lock()
        * acquire_write_lock()
        * release_read_lock()
        * release_write_lock()

    The ``__init__()`` method accepts a single argument "filename", which
    may be used as the "lock file", for those implementations that use a lock
    file.

    Note that multithreaded environments must provide a thread-safe
    version of this lock.  The recommended approach for file-
    descriptor-based locks is to use a Python ``threading.local()`` so
    that a unique file descriptor is held per thread.  See the source
    code of :class:`.FileLock` for an implementation example.


    """

    def __init__(self, filename):
        """Constructor, is given the filename of a potential lockfile.

        The usage of this filename is optional and no file is
        created by default.

        Raises ``NotImplementedError`` by default, must be
        implemented by subclasses.
        """
        raise NotImplementedError()

    def acquire(self, wait=True):
        """Acquire the "write" lock.

        This is a direct call to :meth:`.AbstractFileLock.acquire_write_lock`.

        """
        return self.acquire_write_lock(wait)

    def release(self):
        """Release the "write" lock.

        This is a direct call to :meth:`.AbstractFileLock.release_write_lock`.

        """
        self.release_write_lock()

    @contextmanager
    def read(self):
        """Provide a context manager for the "read" lock.

        This method makes use of :meth:`.AbstractFileLock.acquire_read_lock`
        and :meth:`.AbstractFileLock.release_read_lock`

        """
        self.acquire_read_lock(True)
        try:
            yield
        finally:
            self.release_read_lock()

    @contextmanager
    def write(self):
        """Provide a context manager for the "write" lock.

        This method makes use of :meth:`.AbstractFileLock.acquire_write_lock`
        and :meth:`.AbstractFileLock.release_write_lock`

        """
        self.acquire_write_lock(True)
        try:
            yield
        finally:
            self.release_write_lock()

    @property
    def is_open(self):
        """optional method."""
        raise NotImplementedError()

    def acquire_read_lock(self, wait):
        """Acquire a 'reader' lock.

        Raises ``NotImplementedError`` by default, must be
        implemented by subclasses.
        """
        raise NotImplementedError()

    def acquire_write_lock(self, wait):
        """Acquire a 'write' lock.

        Raises ``NotImplementedError`` by default, must be
        implemented by subclasses.
        """
        raise NotImplementedError()

    def release_read_lock(self):
        """Release a 'reader' lock.

        Raises ``NotImplementedError`` by default, must be
        implemented by subclasses.
        """
        raise NotImplementedError()

    def release_write_lock(self):
        """Release a 'writer' lock.

        Raises ``NotImplementedError`` by default, must be
        implemented by subclasses.
        """
        raise NotImplementedError()