from functools import wraps
def _synchPre(self):
    if '_threadable_lock' not in self.__dict__:
        _synchLockCreator.acquire()
        if '_threadable_lock' not in self.__dict__:
            self.__dict__['_threadable_lock'] = XLock()
        _synchLockCreator.release()
    self._threadable_lock.acquire()