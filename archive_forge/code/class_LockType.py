class LockType(object):
    """Class implementing dummy implementation of _thread.LockType.

    Compatibility is maintained by maintaining self.locked_status
    which is a boolean that stores the state of the lock.  Pickling of
    the lock, though, should not be done since if the _thread module is
    then used with an unpickled ``lock()`` from here problems could
    occur from this class not having atomic methods.

    """

    def __init__(self):
        self.locked_status = False

    def acquire(self, waitflag=None, timeout=-1):
        """Dummy implementation of acquire().

        For blocking calls, self.locked_status is automatically set to
        True and returned appropriately based on value of
        ``waitflag``.  If it is non-blocking, then the value is
        actually checked and not set if it is already acquired.  This
        is all done so that threading.Condition's assert statements
        aren't triggered and throw a little fit.

        """
        if waitflag is None or waitflag:
            self.locked_status = True
            return True
        elif not self.locked_status:
            self.locked_status = True
            return True
        else:
            if timeout > 0:
                import time
                time.sleep(timeout)
            return False
    __enter__ = acquire

    def __exit__(self, typ, val, tb):
        self.release()

    def release(self):
        """Release the dummy lock."""
        if not self.locked_status:
            raise error
        self.locked_status = False
        return True

    def locked(self):
        return self.locked_status