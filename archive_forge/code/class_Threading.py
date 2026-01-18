import atexit
import queue
import threading
import weakref
class Threading(object):

    @staticmethod
    def event_object(*args, **kwargs):
        return threading.Event(*args, **kwargs)

    @staticmethod
    def lock_object(*args, **kwargs):
        return threading.Lock(*args, **kwargs)

    @staticmethod
    def rlock_object(*args, **kwargs):
        return threading.RLock(*args, **kwargs)

    @staticmethod
    def condition_object(*args, **kwargs):
        return threading.Condition(*args, **kwargs)