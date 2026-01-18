import logging
import weakref
from threading import local as thread_local
from threading import Event
from threading import Thread
from peewee import __deprecated__
from playhouse.sqlite_ext import SqliteExtDatabase
class GreenletHelper(ThreadHelper):
    __slots__ = ()

    def event(self):
        return GEvent()

    def queue(self, max_size=None):
        max_size = max_size if max_size is not None else self.queue_max_size
        return GQueue(maxsize=max_size or 0)

    def thread(self, fn, *args, **kwargs):

        def wrap(*a, **k):
            gevent.sleep()
            return fn(*a, **k)
        return GThread(wrap, *args, **kwargs)