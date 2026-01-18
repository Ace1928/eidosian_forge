import sys
import types
import stackless
class greenlet:

    def __init__(self, run=None, parent=None):
        self.dead = False
        if parent is None:
            parent = getcurrent()
        self.parent = parent
        if run is not None:
            self.run = run
        self.switch = FirstSwitch(self)

    def switch(self, *args):
        global caller
        caller = stackless.getcurrent()
        coro_args[self] = args
        self.t.insert()
        stackless.schedule()
        if caller is not self.t:
            caller.remove()
        rval = coro_args[self]
        return rval

    def run(self):
        pass

    def __bool__(self):
        return self.run is None and (not self.dead)