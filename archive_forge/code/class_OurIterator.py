import cherrypy
from cherrypy.test import helper
class OurIterator(IteratorBase):
    started = False
    closed_off = False
    count = 0

    def increment(self):
        self.incr()

    def decrement(self):
        if not self.closed_off:
            self.closed_off = True
            self.decr()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.started:
            self.started = True
            self.increment()
        self.count += 1
        if self.count > 1024:
            raise StopIteration
        return self.datachunk
    next = __next__

    def __del__(self):
        self.decrement()