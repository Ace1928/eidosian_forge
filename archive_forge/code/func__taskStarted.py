import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
def _taskStarted(self, pid, i, **kwds):
    if self.showProgress:
        if len(self.progress[pid]) > 0:
            self.progressDlg += 1
        if pid == os.getpid():
            if self.progressDlg.wasCanceled():
                raise CanceledError()
    self.progress[pid].append(i)