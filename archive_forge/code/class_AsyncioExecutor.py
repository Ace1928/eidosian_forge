from __future__ import absolute_import
from asyncio import Future, get_event_loop, iscoroutine, wait
from wandb_promise import Promise
class AsyncioExecutor(object):

    def __init__(self, loop=None):
        if loop is None:
            loop = get_event_loop()
        self.loop = loop
        self.futures = []

    def wait_until_finished(self):
        while self.futures:
            futures = self.futures
            self.futures = []
            self.loop.run_until_complete(wait(futures))

    def execute(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, Future) or iscoroutine(result):
            future = ensure_future(result, loop=self.loop)
            self.futures.append(future)
            return Promise.resolve(future)
        return result