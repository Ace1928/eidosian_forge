from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class new_style_value(object):

    def measure(self):
        global t0
        self.init()
        if t0 < 0.0:
            t0 = measure(self.null)
        t1 = measure(self.do_get)
        t2 = measure(self.do_set)
        scale = 1000000.0 / n
        get_time = max(t1 - t0, 0.0) * scale
        set_time = max(t2 - t0, 0.0) * scale
        return (get_time, set_time)

    def null(self):
        for i in range(n):
            pass

    def init(self):
        self.value = -1

    def do_set(self):
        for i in range(n):
            self.value = i

    def do_get(self):
        for i in range(n):
            self.value