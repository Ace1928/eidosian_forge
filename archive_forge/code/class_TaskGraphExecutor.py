from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import functools
import multiprocessing
import sys
import threading
from googlecloudsdk.api_lib.storage.gcs_json import patch_apitools_messages
from googlecloudsdk.command_lib import crash_handling
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_buffer
from googlecloudsdk.command_lib.storage.tasks import task_graph as task_graph_module
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds_context_managers
from googlecloudsdk.core.util import platforms
from six.moves import queue
class TaskGraphExecutor:
    """Executes an iterable of command_lib.storage.tasks.task.Task instances."""

    def __init__(self, task_iterator, max_process_count=multiprocessing.cpu_count(), thread_count=4, task_status_queue=None, progress_manager_args=None):
        """Initializes a TaskGraphExecutor instance.

    No threads or processes are started by the constructor.

    Args:
      task_iterator (Iterable[command_lib.storage.tasks.task.Task]): Task
        instances to execute.
      max_process_count (int): The number of processes to start.
      thread_count (int): The number of threads to start per process.
      task_status_queue (multiprocessing.Queue|None): Used by task to report its
        progress to a central location.
      progress_manager_args (task_status.ProgressManagerArgs|None):
        Determines what type of progress indicator to display.
    """
        self._task_iterator = iter(task_iterator)
        self._max_process_count = max_process_count
        self._thread_count = thread_count
        self._task_status_queue = task_status_queue
        self._progress_manager_args = progress_manager_args
        self._process_count = 0
        self._idle_thread_count = multiprocessing_context.Semaphore(value=0)
        self._worker_count = self._max_process_count * self._thread_count
        self._task_queue = multiprocessing_context.Queue(maxsize=1)
        self._task_output_queue = multiprocessing_context.Queue(maxsize=self._worker_count)
        self._signal_queue = multiprocessing_context.Queue(maxsize=self._worker_count + 1)
        self._task_graph = task_graph_module.TaskGraph(top_level_task_limit=2 * self._worker_count)
        self._executable_tasks = task_buffer.TaskBuffer()
        self.thread_exception = None
        self.thread_exception_lock = threading.Lock()
        self._accepting_new_tasks = True
        self._exit_code = 0

    def _add_worker_process(self):
        """Signal the worker process spawner to create a new process."""
        self._signal_queue.put(_CREATE_WORKER_PROCESS)
        self._process_count += 1

    @_store_exception
    def _get_tasks_from_iterator(self):
        """Adds tasks from self._task_iterator to the executor.

    This involves adding tasks to self._task_graph, marking them as submitted,
    and adding them to self._executable_tasks.
    """
        while self._accepting_new_tasks:
            try:
                task_object = next(self._task_iterator)
            except StopIteration:
                break
            task_wrapper = self._task_graph.add(task_object)
            if task_wrapper is None:
                continue
            task_wrapper.is_submitted = True
            self._executable_tasks.put(task_wrapper, prioritize=False)

    @_store_exception
    def _add_executable_tasks_to_queue(self):
        """Sends executable tasks to consumer threads in child processes."""
        task_wrapper = None
        while True:
            if task_wrapper is None:
                task_wrapper = self._executable_tasks.get()
                if task_wrapper == _SHUTDOWN:
                    break
            reached_process_limit = self._process_count >= self._max_process_count
            try:
                self._task_queue.put(task_wrapper, block=reached_process_limit)
                task_wrapper = None
            except queue.Full:
                if self._idle_thread_count.acquire(block=False):
                    self._idle_thread_count.release()
                else:
                    self._add_worker_process()

    @_store_exception
    def _handle_task_output(self):
        """Updates a dependency graph based on information from executed tasks."""
        while True:
            output = self._task_output_queue.get()
            if output == _SHUTDOWN:
                break
            executed_task_wrapper, task_output = output
            if task_output and task_output.messages:
                for message in task_output.messages:
                    if message.topic in (task.Topic.CHANGE_EXIT_CODE, task.Topic.FATAL_ERROR):
                        self._exit_code = 1
                        if message.topic == task.Topic.FATAL_ERROR:
                            self._accepting_new_tasks = False
            submittable_tasks = self._task_graph.update_from_executed_task(executed_task_wrapper, task_output)
            for task_wrapper in submittable_tasks:
                task_wrapper.is_submitted = True
                self._executable_tasks.put(task_wrapper)

    @contextlib.contextmanager
    def _get_worker_process_spawner(self, shared_process_context):
        """Creates a worker process spawner.

    Must be used as a context manager since the worker process spawner must be
    non-daemonic in order to start child processes, but non-daemonic child
    processes block parent processes from exiting, so if there are any failures
    after the worker process spawner is started, gcloud storage will fail to
    exit, unless we put the shutdown logic in a `finally` block.

    Args:
      shared_process_context (SharedProcessContext): Holds values from global
        state that need to be replicated in child processes.

    Yields:
      None, allows body of a `with` statement to execute.
    """
        worker_process_spawner = multiprocessing_context.Process(target=_process_factory, args=(self._task_queue, self._task_output_queue, self._task_status_queue, self._thread_count, self._idle_thread_count, self._signal_queue, shared_process_context))
        try:
            worker_process_spawner.start()
            yield
        finally:
            self._signal_queue.put(_SHUTDOWN)
            worker_process_spawner.join()

    def run(self):
        """Executes tasks from a task iterator in parallel.

    Returns:
      An integer indicating the exit code. Zero indicates no fatal errors were
        raised.
    """
        shared_process_context = SharedProcessContext()
        with self._get_worker_process_spawner(shared_process_context):
            with task_status.progress_manager(self._task_status_queue, self._progress_manager_args):
                self._add_worker_process()
                get_tasks_from_iterator_thread = threading.Thread(target=self._get_tasks_from_iterator)
                add_executable_tasks_to_queue_thread = threading.Thread(target=self._add_executable_tasks_to_queue)
                handle_task_output_thread = threading.Thread(target=self._handle_task_output)
                get_tasks_from_iterator_thread.start()
                add_executable_tasks_to_queue_thread.start()
                handle_task_output_thread.start()
                get_tasks_from_iterator_thread.join()
                try:
                    self._task_graph.is_empty.wait()
                except console_io.OperationCancelledError:
                    pass
                self._executable_tasks.put(_SHUTDOWN)
                self._task_output_queue.put(_SHUTDOWN)
                handle_task_output_thread.join()
                add_executable_tasks_to_queue_thread.join()
        self._task_queue.close()
        self._task_output_queue.close()
        with self.thread_exception_lock:
            if self.thread_exception:
                raise self.thread_exception
        return self._exit_code