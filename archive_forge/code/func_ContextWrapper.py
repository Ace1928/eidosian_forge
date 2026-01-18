from eventlet.patcher import slurp_properties
import sys
import functools
from eventlet import greenthread
from eventlet import patcher
import _thread
@functools.wraps(f)
def ContextWrapper(self, arg, t):
    current = greenthread.getcurrent()
    if current != self.current_tasklet:
        self.SwitchTasklet(self.current_tasklet, current, t)
        t = 0.0
    return f(self, arg, t)