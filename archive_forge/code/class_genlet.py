from greenlet import greenlet
from . import TestCase
class genlet(greenlet):
    parent = None

    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

    def run(self):
        fn, = self.fn
        fn(*self.args, **self.kwds)

    def __iter__(self):
        return self

    def __next__(self):
        self.parent = greenlet.getcurrent()
        result = self.switch()
        if self:
            return result
        raise StopIteration
    next = __next__