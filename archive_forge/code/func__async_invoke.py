from collections import deque
from threading import local
def _async_invoke(self, fn, scheduler):
    self.normal_queue.append(fn)
    self.queue_tick(scheduler)