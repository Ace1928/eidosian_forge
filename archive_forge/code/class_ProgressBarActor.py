import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
@ray.remote
class ProgressBarActor:

    def __init__(self):
        self._init()

    def submit(self, key, deps, now):
        for dep in deps.keys():
            self.deps[key].add(dep)
        self.submitted[key] = now
        self.submission_queue.append((key, now))

    def task_scheduled(self, key, now):
        self.scheduled[key] = now

    def finish(self, key, now):
        self.finished[key] = now

    def result(self):
        return (len(self.submitted), len(self.finished))

    def report(self):
        result = defaultdict(dict)
        for key, finished in self.finished.items():
            submitted = self.submitted[key]
            scheduled = self.scheduled[key]
            result[key]['execution_time'] = (finished - scheduled).total_seconds()
            result[key]['scheduling_time'] = (scheduled - submitted).total_seconds()
        result['submission_order'] = self.submission_queue
        return result

    def ready(self):
        pass

    def reset(self):
        self._init()

    def _init(self):
        self.submission_queue = []
        self.submitted = defaultdict(None)
        self.scheduled = defaultdict(None)
        self.finished = defaultdict(None)
        self.deps = defaultdict(set)