from __future__ import (absolute_import, division, print_function)
import os
import sys
import traceback
from jinja2.exceptions import TemplateNotFound
from multiprocessing.queues import Queue
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.executor.task_executor import TaskExecutor
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
from ansible.utils.multiprocessing import context as multiprocessing_context
class WorkerProcess(multiprocessing_context.Process):
    """
    The worker thread class, which uses TaskExecutor to run tasks
    read from a job queue and pushes results into a results queue
    for reading later.
    """

    def __init__(self, final_q, task_vars, host, task, play_context, loader, variable_manager, shared_loader_obj, worker_id):
        super(WorkerProcess, self).__init__()
        self._final_q = final_q
        self._task_vars = task_vars
        self._host = host
        self._task = task
        self._play_context = play_context
        self._loader = loader
        self._variable_manager = variable_manager
        self._shared_loader_obj = shared_loader_obj
        self._loader._tempfiles = set()
        self.worker_queue = WorkerQueue(ctx=multiprocessing_context)
        self.worker_id = worker_id

    def _save_stdin(self):
        self._new_stdin = None
        try:
            if sys.stdin.isatty() and sys.stdin.fileno() is not None:
                try:
                    self._new_stdin = os.fdopen(os.dup(sys.stdin.fileno()))
                except OSError:
                    pass
        except (AttributeError, ValueError):
            pass
        if self._new_stdin is None:
            self._new_stdin = open(os.devnull)

    def start(self):
        """
        multiprocessing.Process replaces the worker's stdin with a new file
        but we wish to preserve it if it is connected to a terminal.
        Therefore dup a copy prior to calling the real start(),
        ensuring the descriptor is preserved somewhere in the new child, and
        make sure it is closed in the parent when start() completes.
        """
        self._save_stdin()
        with display._lock:
            try:
                return super(WorkerProcess, self).start()
            finally:
                self._new_stdin.close()

    def _hard_exit(self, e):
        """
        There is no safe exception to return to higher level code that does not
        risk an innocent try/except finding itself executing in the wrong
        process. All code executing above WorkerProcess.run() on the stack
        conceptually belongs to another program.
        """
        try:
            display.debug(u'WORKER HARD EXIT: %s' % to_text(e))
        except BaseException:
            pass
        os._exit(1)

    def run(self):
        """
        Wrap _run() to ensure no possibility an errant exception can cause
        control to return to the StrategyBase task loop, or any other code
        higher in the stack.

        As multiprocessing in Python 2.x provides no protection, it is possible
        a try/except added in far-away code can cause a crashed child process
        to suddenly assume the role and prior state of its parent.
        """
        try:
            return self._run()
        except BaseException as e:
            self._hard_exit(e)
        finally:
            sys.stdout = sys.stderr = open(os.devnull, 'w')

    def _run(self):
        """
        Called when the process is started.  Pushes the result onto the
        results queue. We also remove the host from the blocked hosts list, to
        signify that they are ready for their next task.
        """
        display.set_queue(self._final_q)
        global current_worker
        current_worker = self
        try:
            display.debug('running TaskExecutor() for %s/%s' % (self._host, self._task))
            executor_result = TaskExecutor(self._host, self._task, self._task_vars, self._play_context, self._new_stdin, self._loader, self._shared_loader_obj, self._final_q, self._variable_manager).run()
            display.debug('done running TaskExecutor() for %s/%s [%s]' % (self._host, self._task, self._task._uuid))
            self._host.vars = dict()
            self._host.groups = []
            display.debug('sending task result for task %s' % self._task._uuid)
            try:
                self._final_q.send_task_result(self._host.name, self._task._uuid, executor_result, task_fields=self._task.dump_attrs())
            except Exception as e:
                display.debug(f'failed to send task result ({e}), sending surrogate result')
                self._final_q.send_task_result(self._host.name, self._task._uuid, {'failed': True, 'msg': f'{e}', 'exception': traceback.format_exc()}, {})
            display.debug('done sending task result for task %s' % self._task._uuid)
        except AnsibleConnectionFailure:
            self._host.vars = dict()
            self._host.groups = []
            self._final_q.send_task_result(self._host.name, self._task._uuid, dict(unreachable=True), task_fields=self._task.dump_attrs())
        except Exception as e:
            if not isinstance(e, (IOError, EOFError, KeyboardInterrupt, SystemExit)) or isinstance(e, TemplateNotFound):
                try:
                    self._host.vars = dict()
                    self._host.groups = []
                    self._final_q.send_task_result(self._host.name, self._task._uuid, dict(failed=True, exception=to_text(traceback.format_exc()), stdout=''), task_fields=self._task.dump_attrs())
                except Exception:
                    display.debug(u'WORKER EXCEPTION: %s' % to_text(e))
                    display.debug(u'WORKER TRACEBACK: %s' % to_text(traceback.format_exc()))
                finally:
                    self._clean_up()
        display.debug('WORKER PROCESS EXITING')

    def _clean_up(self):
        self._loader.cleanup_all_tmp_files()