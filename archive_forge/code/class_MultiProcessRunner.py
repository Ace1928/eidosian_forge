import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
class MultiProcessRunner(object):
    """A utility class to start multiple processes to simulate a cluster.

  We need to use multiple processes to simulate a cluster in TF 2.0 tests
  because TF 2.0 has some process-global data structures that have to be
  separated by processes. We also need child processes to test out our fault
  tolerance because shutting down a standard TensorFlow server within its
  process is not supported.

  Note: the main test program that uses this runner class must run main program
  via `test_main` defined in this file. Using this runner in non-test binaries
  is not supported yet.

  This class is not thread-safe. Child processes will inherit TF2 behavior flag.
  """

    def __init__(self, fn, cluster_spec, rpc_layer=None, max_run_time=None, grpc_fail_fast=None, stream_output=True, return_output=False, use_dill_for_args=True, daemon=False, dependence_on_chief=True, auto_restart=False, share_gpu=True, args=None, kwargs=None):
        """Instantiation of a `MultiProcessRunner`.

    Args:
      fn: Function to be run on child processes. This will be run on processes
        for all task types.
      cluster_spec: Dict for cluster spec. The utility function
        `tf.__internal__.distribute.multi_process_runner.create_cluster_spec`
        can be conveniently used to create such dict. The following is an
        example of cluster with three workers and two ps's.
        {"worker": ["worker0.example.com:2222",
                    "worker1.example.com:2222",
                    "worker2.example.com:2222"],
         "ps": ["ps0.example.com:2222",
                "ps1.example.com:2222"]}
      rpc_layer: RPC layer to use. Default value is 'grpc'.
      max_run_time: `None` or integer. If not `None`, child processes are forced
        to exit at approximately this many seconds after this utility is called.
        We achieve this through `signal.alarm()` api. Note that this is best
        effort at Python level since Python signal handler does not get executed
        when it runs lower level C/C++ code. So it can be delayed for
        arbitrarily long time. If any of the child process is still running when
        `max_run_time` is up, they will be force-terminated and an
        `UnexpectedSubprocessExitError` may be raised. If `None`, child
        processes are not forced to exit.
      grpc_fail_fast: Whether GRPC connection between processes should fail
        without retrying. Defaults to None, in which case the environment
        variable is not explicitly set.
      stream_output: True if the output/error from the subprocesses should be
        streamed to be printed in parent process' log. Defaults to True.
      return_output: If True, the output/error from the subprocesses should be
        collected to be attached to the resulting namedtuple returned from
        `join()`. The list of output can be retrieved via `stdout` attribute.
        Defaults to False.
      use_dill_for_args: Whether to use dill to pickle `args` and `kwargs`. dill
        can pickle more objects, but doesn't work with types in
        `multiprocessing` library like `Mutex`.
      daemon: Whether to start processes as daemons.
      dependence_on_chief: Whether to terminates the cluster if the chief exits.
        If auto_restart is True, it only terminates the cluster if the chief
        exits with a zero exit code.
      auto_restart: Whether to automatically restart processes that exit with
        non-zero exit code.
      share_gpu: Whether to share GPUs among workers. If False, each worker is
        assigned different GPUs in a roundrobin fashion. This should be True
        whenever possible for better test execution coverage; some situations
        that need it to be False are tests that runs NCCL.
      args: Positional arguments to be sent to `fn` run on subprocesses.
      kwargs: Keyword arguments to be sent to `fn` run on subprocesses.

    Raises:
      RuntimeError: if `multi_process_runner.test_main()` is not called.
      ValueError: if there are more than one chief in the `cluster_spec`.
      SkipTest: if thread sanitizer is enabled (which is incompatible with MPR).
    """
        if test_util.is_tsan_enabled():
            raise unittest.SkipTest('ThreadSanitizer is not compatible with MultiProcessRunner.')
        assert cluster_spec is not None
        if 'chief' in cluster_spec and len(cluster_spec['chief']) > 1:
            raise ValueError('If chief exists in the cluster, there must be at most one chief. Current `cluster_spec` has {} chiefs.'.format(len(cluster_spec['chief'])))
        _check_initialization()
        if not callable(fn):
            raise ValueError('fn is not a callable')
        self._fn = fn
        self._cluster_spec = cluster_spec
        self._rpc_layer = rpc_layer or 'grpc'
        self._max_run_time = max_run_time
        self._grpc_fail_fast = grpc_fail_fast
        self._stream_output = stream_output
        self._return_output = return_output
        self._dependence_on_chief = dependence_on_chief
        self._use_dill_for_args = use_dill_for_args
        self._daemon = daemon
        self._auto_restart = auto_restart
        self._args = args or ()
        self._kwargs = kwargs or {}
        self._share_gpu = share_gpu
        self._total_gpu = len(context.context().list_physical_devices('GPU'))
        self._v2_enabled = tf2.enabled()
        self._executing_eagerly = context.executing_eagerly()
        self._joined = False
        self._process_lock = threading.Lock()
        self._processes = {}
        self._terminated = set()
        self._reading_threads = []
        self._manager = manager()
        self._process_status_queue = self._manager.Queue()
        self._parent_to_sub_queue = self._manager.Queue()
        parties = sum((len(addresses) for addresses in self._cluster_spec.values()))
        self._barrier = self._manager.Barrier(parties)
        self._streaming_queue = self._manager.Queue()
        self._watchdog_thread = None

    def set_args(self, args=None, kwargs=None):
        self._args = args or self._args
        self._kwargs = kwargs or self._kwargs

    def _continuously_readline_from_sub(self, pipe_r, task_type, task_id):
        """Function to continuously read lines from subprocesses."""
        with os.fdopen(pipe_r.fileno(), 'r', closefd=False) as reader:
            for line in reader:
                task_string = '[{}-{}]:'.format(task_type, task_id)
                formatted_line = '{} {}'.format(task_string.ljust(14), line)
                if self._stream_output:
                    print(formatted_line, end='', flush=True)
                if self._return_output:
                    self._streaming_queue.put(formatted_line)

    def _start_subprocess_and_reading_thread(self, task_type, task_id, cluster_spec=None, fn=None, args=None, kwargs=None):
        """Start a subprocess and a thread the reads lines from the subprocess."""
        if dill is None:
            raise unittest.SkipTest('TODO(b/150264776): Resolve dependency issue in CI')
        cluster_spec = cluster_spec or self._cluster_spec
        visible_gpus = None
        if not self._share_gpu and self._total_gpu > 0:
            id_in_cluster = multi_worker_util.id_in_cluster(cluster_spec, task_type, task_id)
            worker_count = multi_worker_util.worker_count(cluster_spec, task_type)
            visible_gpus = list(range(id_in_cluster, self._total_gpu, worker_count))
        test_env = TestEnvironment(task_type=task_type, task_id=task_id, cluster_spec=cluster_spec, rpc_layer=self._rpc_layer, grpc_fail_fast=self._grpc_fail_fast, v2_enabled=self._v2_enabled, executing_eagerly=self._executing_eagerly, visible_gpus=visible_gpus)
        pipe_r, pipe_w = multiprocessing.Pipe(duplex=False)
        resources = Resources(process_status_queue=self._process_status_queue, parent_to_sub_queue=self._parent_to_sub_queue, streaming_pipe_w=pipe_w, barrier=self._barrier)
        if fn is None:
            fn, args, kwargs = (self._fn, self._args, self._kwargs)
        fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
        if self._use_dill_for_args:
            args = dill.dumps(args, dill.HIGHEST_PROTOCOL)
            kwargs = dill.dumps(kwargs, dill.HIGHEST_PROTOCOL)
        p = _Process(test_env=test_env, target=_ProcFunc(), args=(resources, test_env, fn, args, kwargs, self._use_dill_for_args), daemon=self._daemon)
        p.start()
        self._processes[task_type, task_id] = p
        self._terminated.discard((task_type, task_id))
        thread = threading.Thread(target=self._continuously_readline_from_sub, args=(pipe_r, task_type, task_id))
        thread.start()
        self._reading_threads.append(thread)
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(target=self._process_watchdog)
            self._watchdog_thread.start()

    def start(self):
        """Starts processes, one for each task in `cluster_spec`.

    Note that this is best effort by the applicable multiprocessing library,
    and it may take up to seconds for a subprocess to be successfully started.
    """
        with self._process_lock:
            if self._processes:
                raise ValueError('MultiProcessRunner already started.')
            if self._joined:
                raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
            for task_type, addresses in self._cluster_spec.items():
                for task_id, _ in enumerate(addresses):
                    self._start_subprocess_and_reading_thread(task_type, task_id)
        if self._max_run_time is not None:

            def handler(signum, frame):
                del signum, frame
                self.terminate_all()
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self._max_run_time)

    def start_in_process_as(self, as_task_type, as_task_id):
        """Start the processes, with the specified task run in main process.

    This is similar to `start()` except that the task with task_type
    `as_task_type` and task_id `as_task_id` is run in the main process.
    This method is particularly useful when debugging tool such as `pdb` is
    needed in some specific task. Note that since this method is blocking until
    that specific task exits, additional actions would need a thread to be
    called:

    ```python
    def fn():
      # user code to be run
      import pdb; pdb.set_trace()

    def follow_ups():
      time.sleep(5)
      mpr.start_single_process(
          task_type='evaluator',
          task_id=0)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1))
    threading.Thread(target=follow_ups).start()
    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
    mpr.join()
    ```

    Note that if `return_output=True`, the logs/stdout by task
    run by the main process is not available in result.stdout.

    Args:
      as_task_type: The task type to be run in the main process.
      as_task_id: The task id to be run in the main process.
    """
        if self._processes:
            raise ValueError('MultiProcessRunner already started.')
        with self._process_lock:
            if self._joined:
                raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
            for task_type, addresses in self._cluster_spec.items():
                for task_id, _ in enumerate(addresses):
                    if not (task_type == as_task_type and task_id == as_task_id):
                        self._start_subprocess_and_reading_thread(task_type, task_id)
        _set_tf_config(as_task_type, as_task_id, self._cluster_spec, self._rpc_layer)
        self._fn(*self._args, **self._kwargs)

    def start_single_process(self, task_type, task_id, cluster_spec=None, fn=None, args=None, kwargs=None):
        """Starts a single process.

    This starts a process in the cluster with the task type, task id, and the
    process function (`fn`). If process function is `None`, the function
    provided at `__init__` will be used. If `cluster_spec` is `None`, the
    cluster spec provided at `__init__` will be used.

    TODO(rchao): It is meant that all subprocesses will be updated with the new
    cluster spec, but this has yet to be implemented. At this time only the
    newly started subprocess picks up this updated cluster spec.

    Args:
      task_type: The task type.
      task_id: The task id.
      cluster_spec: The cluster spec to be used on the newly started
        process. If `None`, the cluster spec provided at `__init__` will be
        used.
      fn: The process function to be run on the newly started
        process. If specified, specify `args` and `kwargs` as well. If `None`,
        the function provided at `__init__` will be used.
      args: Optional positional arguments to be supplied in `fn`.
      kwargs: Optional keyword arguments to be supplied in `fn`.
    """
        with self._process_lock:
            if self._joined:
                raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
            self._start_subprocess_and_reading_thread(task_type, task_id, cluster_spec=cluster_spec, fn=fn, args=args or (), kwargs=kwargs or {})

    def _queue_to_list(self, queue_to_convert):
        """Convert `queue.Queue` to `list`."""
        list_to_return = []
        while True:
            try:
                list_to_return.append(queue_to_convert.get(block=False))
            except Queue.Empty:
                break
        return list_to_return

    def _get_process_statuses(self):
        statuses = {}
        for status in self._queue_to_list(self._process_status_queue):
            statuses[status.task_type, status.task_id] = status
        return statuses

    def get_process_id(self, task_type, task_id):
        """Returns the subprocess id given the task type and task id."""
        with self._process_lock:
            p = self._processes.get((task_type, task_id), None)
        return p.pid if p else None

    def get_process_exit_code(self, task_type, task_id):
        """Returns the subprocess exit code given the task type and task id.

    Args:
      task_type: The task type.
      task_id: The task id.

    Returns:
      The subprocess exit code; `None` if the subprocess has not exited yet.

    Raises:
      KeyError: If the corresponding subprocess is not found with `task_type`
        and `task_id`.
    """
        with self._process_lock:
            p = self._processes[task_type, task_id]
        return p.exitcode if p else None

    def process_exists(self, task_type, task_id):
        """Returns whether the subprocess still exists given the task type and id.

    Args:
      task_type: The task type.
      task_id: The task id.

    Returns:
      Boolean; whether the subprocess still exists. If the subprocess has
      exited, this returns False.
    """
        return self.get_process_exit_code(task_type, task_id) is None

    def _process_watchdog(self):
        """Simulates a cluster management system.

    - If auto_restart is True, it restarts processes that exit with a non-zero
      exit code. Note that when join() times out it overrides auto_restart to
      False.
    - If dependence_on_chief is True, it terminates all processes once the chief
      exits. If auto_restart is also True, it only terminates all processes if
      the chief exit with a zero exit code, otherwise it restarts the chief.

    This runs in self._watchdog_thread.
    """
        while True:
            time.sleep(1)
            with self._process_lock:
                chief = self._processes.get(('chief', 0), None)
                if chief and self._dependence_on_chief and (chief.exitcode is not None):
                    if chief.exitcode == 0 or not self._auto_restart:
                        for p in self._processes.values():
                            p.join(timeout=3)
                        self._terminate_all()
                        for p in self._processes.values():
                            p.join()
                        return
                if self._auto_restart:
                    has_failure = False
                    for (task_type, task_id), p in self._processes.items():
                        if p.exitcode is not None and p.exitcode != 0:
                            has_failure = True
                            logging.info('Restarting failed %s-%d', task_type, task_id)
                            self._start_subprocess_and_reading_thread(task_type, task_id)
                    if has_failure:
                        continue
                if all((p.exitcode is not None for p in self._processes.values())):
                    return

    def _reraise_if_subprocess_error(self, process_statuses):
        for process_status in process_statuses.values():
            assert isinstance(process_status, _ProcessStatusInfo)
            if not process_status.is_successful:
                process_status.exc_info[1].mpr_result = self._get_mpr_result(process_statuses)
                six.reraise(*process_status.exc_info)

    def join(self, timeout=_DEFAULT_TIMEOUT_SEC):
        """Joins all the processes with timeout.

    If any of the subprocesses does not exit approximately after `timeout`
    seconds has passed after `join` call, this raises a
    `SubprocessTimeoutError`.

    Note: At timeout, it uses SIGTERM to terminate the subprocesses, in order to
    log the stack traces of the subprocesses when they exit. However, this
    results in timeout when the test runs with tsan (thread sanitizer); if tsan
    is being run on the test targets that rely on timeout to assert information,
    `MultiProcessRunner.terminate_all()` must be called after `join()`, before
    the test exits, so the subprocesses are terminated with SIGKILL, and data
    race is removed.

    Args:
      timeout: optional integer or `None`. If provided as an integer, and not
      all processes report status within roughly `timeout` seconds, a
      `SubprocessTimeoutError` exception will be raised. If `None`, `join` never
      times out.

    Returns:
      A `MultiProcessRunnerResult` object, which has two attributes,
      `return_value` and `stdout`. `return_value` always contains a list of
      return values from the subprocesses, although the order is not meaningful.
      If `return_output` argument is True at `__init__`, `stdout` is available
      that contains a list of all messages from subprocesses' stdout and stderr.

    Raises:
      SubprocessTimeoutError: if not all processes report status approximately
        within `timeout` seconds. When this is raised, a
        `MultiProcessRunnerResult` object can be retrieved by
        `SubprocessTimeoutError`'s mpr_result attribute, which has the same
        structure as above 'Returns' section describes.
      UnexpectedSubprocessExitError: If any of the subprocesses did not exit
        properly (for example, they exit on SIGTERM or SIGKILL signal). When
        this is raised, a `MultiProcessRunnerResult` object can be retrieved by
        `UnexpectedSubprocessExitError`'s mpr_result attribute, which has the
        same structure as above 'Returns' section describes. If `max_run_time`
        is not `None`, it is expected that some subprocesses may be
        force-killed when `max_run_time` is up, and this is raised in those
        cases.
      Exception: if there is an Exception propagated from any subprocess. When
        this is raised, a `MultiProcessRunnerResult` object can be retrieved by
        `UnexpectedSubprocessExitError`'s mpr_result attribute, which has the
        same structure as above 'Returns' section describes.
    """
        if timeout and (not isinstance(timeout, int)):
            raise ValueError('`timeout` must be an integer or `None`.')
        with self._process_lock:
            if self._joined:
                raise ValueError("MultiProcessRunner can't be joined twice.")
            self._joined = True
        self._watchdog_thread.join(timeout)
        if self._watchdog_thread.is_alive():
            with self._process_lock:
                self._auto_restart = False
            logging.error('Timeout when joining for child processes. Terminating...')
            self.terminate_all(sig=signal.SIGTERM)
            self._watchdog_thread.join(_FORCE_KILL_WAIT_SEC)
            if self._watchdog_thread.is_alive():
                logging.error('Timeout when waiting for child processes to print stacktrace. Sending SIGKILL...')
                self.terminate_all()
                self._watchdog_thread.join()
            process_statuses = self._get_process_statuses()
            self._reraise_if_subprocess_error(process_statuses)
            raise SubprocessTimeoutError('One or more subprocesses timed out, where timeout was set to {}s. Please change the `timeout` argument for `MultiProcessRunner.join()` or `multi_process_runner.run()` if it should be adjusted.'.format(timeout), self._get_mpr_result(process_statuses))
        for (task_type, task_id), p in self._processes.items():
            logging.info('%s-%d exit code: %s', task_type, task_id, p.exitcode)
        process_statuses = self._get_process_statuses()
        self._reraise_if_subprocess_error(process_statuses)
        for (task_type, task_id), p in self._processes.items():
            assert p.exitcode is not None
            if p.exitcode > 0 and (task_type, task_id) not in self._terminated:
                raise UnexpectedSubprocessExitError('Subprocess %s-%d exited with exit code %s. See logs for details.' % (task_type, task_id, p.exitcode), self._get_mpr_result(process_statuses))
        logging.info('Joining log reading threads.')
        for thread in self._reading_threads:
            thread.join()
        logging.info('Joined log reading threads.')
        signal.alarm(0)
        return self._get_mpr_result(process_statuses)

    def _get_mpr_result(self, process_statuses):
        stdout = self._queue_to_list(self._streaming_queue)
        return_values = []
        for process_status in process_statuses.values():
            if process_status.return_value is not None:
                return_values.append(process_status.return_value)
        return MultiProcessRunnerResult(stdout=stdout, return_value=return_values)

    def terminate(self, task_type, task_id):
        """Terminates the process with `task_type` and `task_id`.

    If auto_retart=True, the terminated task will be restarted unless the chief
    has already exited with zero exit code.

    Args:
      task_type: the task type.
      task_id: the task id.

    """
        with self._process_lock:
            p = self._processes.get((task_type, task_id), None)
            if p is None:
                raise ValueError('{}-{} does not exist'.format(task_type, task_id))
            self._terminated.add((task_type, task_id))
            self._parent_to_sub_queue.put('terminate {} {}'.format(task_type, task_id))
            p.join()

    def _terminate_all(self, sig=None):
        """Terminates all subprocesses.

    The caller is required to hold self._process_lock.

    Args:
      sig: the signal used to terminate the process. The default is SIGKILL.
    """
        sig = sig or getattr(signal, 'SIGKILL', signal.SIGTERM)
        for (task_type, task_id), p in self._processes.items():
            if p.exitcode is not None:
                logging.info('%s-%d has already exited. Not terminating.', task_type, task_id)
                continue
            try:
                os.kill(p.pid, sig)
                self._terminated.add((task_type, task_id))
                logging.info('%s-%d terminated with signal %r.', task_type, task_id, sig)
            except ProcessLookupError:
                logging.info('Attempting to kill %s-%d but it does not exist.', task_type, task_id)

    def terminate_all(self, sig=None):
        """Terminates all subprocesses."""
        with self._process_lock:
            self._terminate_all(sig)