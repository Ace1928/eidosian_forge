from collections import deque
from threading import local
def _async_settle_promise(self, promise):
    self.normal_queue.append(promise)
    self.queue_tick(promise.scheduler)