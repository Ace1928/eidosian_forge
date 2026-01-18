from eventlet.patcher import slurp_properties
import sys
import functools
from eventlet import greenthread
from eventlet import patcher
import _thread
def SwitchTasklet(self, t0, t1, t):
    pt, it, et, fn, frame, rcur = self.cur
    cur = (pt, it + t, et, fn, frame, rcur)
    self.sleeping[t0] = (cur, self.timings)
    self.current_tasklet = t1
    try:
        self.cur, self.timings = self.sleeping.pop(t1)
    except KeyError:
        self.cur, self.timings = (None, {})
        self.simulate_call('profiler')
        self.simulate_call('new_tasklet')