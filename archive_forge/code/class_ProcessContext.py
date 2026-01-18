import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings
from typing import Optional
from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]
class ProcessContext:

    def __init__(self, processes, error_queues):
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {process.sentinel: index for index, process in enumerate(processes)}

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        """Join one or more processes within spawn context.

        Attempt to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        if len(self.sentinels) == 0:
            return True
        ready = multiprocessing.connection.wait(self.sentinels.keys(), timeout=timeout)
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
        if error_index is None:
            return len(self.sentinels) == 0
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException('process %d terminated with signal %s' % (error_index, name), error_index=error_index, error_pid=failed_process.pid, exit_code=exitcode, signal_name=name)
            else:
                raise ProcessExitedException('process %d terminated with exit code %d' % (error_index, exitcode), error_index=error_index, error_pid=failed_process.pid, exit_code=exitcode)
        original_trace = self.error_queues[error_index].get()
        msg = '\n\n-- Process %d terminated with the following error:\n' % error_index
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)