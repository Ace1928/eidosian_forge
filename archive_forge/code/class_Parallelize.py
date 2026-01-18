import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
class Parallelize(object):
    """
    Class for ultra-simple inline parallelization on multi-core CPUs
    
    Example::
    
        ## Here is the serial (single-process) task:
        
        tasks = [1, 2, 4, 8]
        results = []
        for task in tasks:
            result = processTask(task)
            results.append(result)
        print(results)
        
        
        ## Here is the parallelized version:
        
        tasks = [1, 2, 4, 8]
        results = []
        with Parallelize(tasks, workers=4, results=results) as tasker:
            for task in tasker:
                result = processTask(task)
                tasker.results.append(result)
        print(results)
        
        
    The only major caveat is that *result* in the example above must be picklable,
    since it is automatically sent via pipe back to the parent process.
    """

    def __init__(self, tasks=None, workers=None, block=True, progressDialog=None, randomReseed=True, **kwds):
        """
        ===============  ===================================================================
        **Arguments:**
        tasks            list of objects to be processed (Parallelize will determine how to 
                         distribute the tasks). If unspecified, then each worker will receive
                         a single task with a unique id number.
        workers          number of worker processes or None to use number of CPUs in the 
                         system
        progressDialog   optional dict of arguments for ProgressDialog
                         to update while tasks are processed
        randomReseed     If True, each forked process will reseed its random number generator
                         to ensure independent results. Works with the built-in random
                         and numpy.random.
        kwds             objects to be shared by proxy with child processes (they will 
                         appear as attributes of the tasker)
        ===============  ===================================================================
        """
        self.showProgress = False
        if progressDialog is not None:
            self.showProgress = True
            if isinstance(progressDialog, str):
                progressDialog = {'labelText': progressDialog}
            from ..widgets.ProgressDialog import ProgressDialog
            self.progressDlg = ProgressDialog(**progressDialog)
        if workers is None:
            workers = self.suggestedWorkerCount()
        if not hasattr(os, 'fork'):
            workers = 1
        self.workers = workers
        if tasks is None:
            tasks = range(workers)
        self.tasks = list(tasks)
        self.reseed = randomReseed
        self.kwds = kwds.copy()
        self.kwds['_taskStarted'] = self._taskStarted

    def __enter__(self):
        self.proc = None
        if self.workers == 1:
            return self.runSerial()
        else:
            return self.runParallel()

    def __exit__(self, *exc_info):
        if self.proc is not None:
            exceptOccurred = exc_info[0] is not None
            try:
                if exceptOccurred:
                    sys.excepthook(*exc_info)
            finally:
                os._exit(1 if exceptOccurred else 0)
        elif self.showProgress:
            try:
                self.progressDlg.__exit__(None, None, None)
            except Exception:
                pass

    def runSerial(self):
        if self.showProgress:
            self.progressDlg.__enter__()
            self.progressDlg.setMaximum(len(self.tasks))
        self.progress = {os.getpid(): []}
        return Tasker(self, None, self.tasks, self.kwds)

    def runParallel(self):
        self.childs = []
        workers = self.workers
        chunks = [[] for i in range(workers)]
        i = 0
        for i in range(len(self.tasks)):
            chunks[i % workers].append(self.tasks[i])
        for i in range(workers):
            proc = ForkedProcess(target=None, preProxy=self.kwds, randomReseed=self.reseed)
            if not proc.isParent:
                self.proc = proc
                return Tasker(self, proc, chunks[i], proc.forkedProxies)
            else:
                self.childs.append(proc)
        self.progress = dict([(ch.childPid, []) for ch in self.childs])
        try:
            if self.showProgress:
                self.progressDlg.__enter__()
                self.progressDlg.setMaximum(len(self.tasks))
            activeChilds = self.childs[:]
            self.exitCodes = []
            pollInterval = 0.01
            while len(activeChilds) > 0:
                waitingChildren = 0
                rem = []
                for ch in activeChilds:
                    try:
                        n = ch.processRequests()
                        if n > 0:
                            waitingChildren += 1
                    except ClosedError:
                        rem.append(ch)
                        if self.showProgress:
                            self.progressDlg += 1
                for ch in rem:
                    activeChilds.remove(ch)
                    while True:
                        try:
                            pid, exitcode = os.waitpid(ch.childPid, 0)
                            self.exitCodes.append(exitcode)
                            break
                        except OSError as ex:
                            if ex.errno == 4:
                                continue
                            else:
                                raise
                if self.showProgress and self.progressDlg.wasCanceled():
                    for ch in activeChilds:
                        ch.kill()
                    raise CanceledError()
                if waitingChildren > 1:
                    pollInterval *= 0.7
                elif waitingChildren == 0:
                    pollInterval /= 0.7
                pollInterval = max(min(pollInterval, 0.5), 0.0005)
                time.sleep(pollInterval)
        finally:
            if self.showProgress:
                self.progressDlg.__exit__(None, None, None)
            for ch in self.childs:
                ch.join()
        if len(self.exitCodes) < len(self.childs):
            raise Exception('Parallelizer started %d processes but only received exit codes from %d.' % (len(self.childs), len(self.exitCodes)))
        for code in self.exitCodes:
            if code != 0:
                raise Exception('Error occurred in parallel-executed subprocess (console output may have more information).')
        return []

    @staticmethod
    def suggestedWorkerCount():
        if 'linux' in sys.platform:
            try:
                cores = {}
                pid = None
                with open('/proc/cpuinfo') as fd:
                    for line in fd:
                        m = re.match('physical id\\s+:\\s+(\\d+)', line)
                        if m is not None:
                            pid = m.groups()[0]
                        m = re.match('cpu cores\\s+:\\s+(\\d+)', line)
                        if m is not None:
                            cores[pid] = int(m.groups()[0])
                return sum(cores.values())
            except:
                return multiprocessing.cpu_count()
        else:
            return multiprocessing.cpu_count()

    def _taskStarted(self, pid, i, **kwds):
        if self.showProgress:
            if len(self.progress[pid]) > 0:
                self.progressDlg += 1
            if pid == os.getpid():
                if self.progressDlg.wasCanceled():
                    raise CanceledError()
        self.progress[pid].append(i)