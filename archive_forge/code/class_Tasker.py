import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
class Tasker(object):

    def __init__(self, parallelizer, process, tasks, kwds):
        self.proc = process
        self.par = parallelizer
        self.tasks = tasks
        for k, v in kwds.items():
            setattr(self, k, v)

    def __iter__(self):
        for i, task in enumerate(self.tasks):
            self.index = i
            self._taskStarted(os.getpid(), i, _callSync='off')
            yield task
        if self.proc is not None:
            self.proc.close()

    def process(self):
        """
        Process requests from parent.
        Usually it is not necessary to call this unless you would like to 
        receive messages (such as exit requests) during an iteration.
        """
        if self.proc is not None:
            self.proc.processRequests()

    def numWorkers(self):
        """
        Return the number of parallel workers
        """
        return self.par.workers