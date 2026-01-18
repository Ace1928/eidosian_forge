import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
class PeriodicWorker(object):
    """Calls a collection of callables periodically (sleeping as needed...).

    NOTE(harlowja): typically the :py:meth:`.start` method is executed in a
    background thread so that the periodic callables are executed in
    the background/asynchronously (using the defined periods to determine
    when each is called).
    """
    MAX_LOOP_IDLE = 30
    _NO_OP_ARGS = ()
    _NO_OP_KWARGS = {}
    _INITIAL_METRICS = {'runs': 0, 'elapsed': 0, 'elapsed_waiting': 0, 'failures': 0, 'successes': 0, 'requested_stop': False}
    _RESCHEDULE_DELAY = 0.9
    _RESCHEDULE_JITTER = 0.2
    DEFAULT_JITTER = fractions.Fraction(5, 100)
    '\n    Default jitter percentage the built-in strategies (that have jitter\n    support) will use.\n    '
    BUILT_IN_STRATEGIES = {'last_started': (_last_started_strategy, _now_plus_periodicity), 'last_started_jitter': (_add_jitter(DEFAULT_JITTER)(_last_started_strategy), _now_plus_periodicity), 'last_finished': (_last_finished_strategy, _now_plus_periodicity), 'last_finished_jitter': (_add_jitter(DEFAULT_JITTER)(_last_finished_strategy), _now_plus_periodicity), 'aligned_last_finished': (_aligned_last_finished_strategy, _now_plus_periodicity), 'aligned_last_finished_jitter': (_add_jitter(DEFAULT_JITTER)(_aligned_last_finished_strategy), _now_plus_periodicity)}
    '\n    Built in scheduling strategies (used to determine when next to run\n    a periodic callable).\n\n    The first element is the strategy to use after the initial start\n    and the second element is the strategy to use for the initial start.\n\n    These are made somewhat pluggable so that we can *easily* add-on\n    different types later (perhaps one that uses a cron-style syntax\n    for example).\n    '

    @classmethod
    def create(cls, objects, exclude_hidden=True, log=None, executor_factory=None, cond_cls=threading.Condition, event_cls=threading.Event, schedule_strategy='last_started', now_func=utils.now, on_failure=None, args=_NO_OP_ARGS, kwargs=_NO_OP_KWARGS):
        """Automatically creates a worker by analyzing object(s) methods.

        Only picks up methods that have been tagged/decorated with
        the :py:func:`.periodic` decorator (does not match against private
        or protected methods unless explicitly requested to).

        :param objects: the objects to introspect for decorated members
        :type objects: iterable
        :param exclude_hidden: exclude hidden members (ones that start with
                               an underscore)
        :type exclude_hidden: bool
        :param log: logger to use when creating a new worker (defaults
                    to the module logger if none provided), it is currently
                    only used to report callback failures (if they occur)
        :type log: logger
        :param executor_factory: factory callable that can be used to generate
                                 executor objects that will be used to
                                 run the periodic callables (if none is
                                 provided one will be created that uses
                                 the :py:class:`~futurist.SynchronousExecutor`
                                 class)
        :type executor_factory: ExecutorFactory or any callable
        :param cond_cls: callable object that can
                          produce ``threading.Condition``
                          (or compatible/equivalent) objects
        :type cond_cls: callable
        :param event_cls: callable object that can produce ``threading.Event``
                          (or compatible/equivalent) objects
        :type event_cls: callable
        :param schedule_strategy: string to select one of the built-in
                                  strategies that can return the
                                  next time a callable should run
        :type schedule_strategy: string
        :param now_func: callable that can return the current time offset
                         from some point (used in calculating elapsed times
                         and next times to run); preferably this is
                         monotonically increasing
        :type now_func: callable
        :param on_failure: callable that will be called whenever a periodic
                           function fails with an error, it will be provided
                           four positional arguments and one keyword
                           argument, the first positional argument being the
                           callable that failed, the second being the type
                           of activity under which it failed (``IMMEDIATE`` or
                           ``PERIODIC``), the third being the spacing that the
                           callable runs at and the fourth ``exc_info`` tuple
                           of the failure. The keyword argument ``traceback``
                           will also be provided that may be be a string
                           that caused the failure (this is required for
                           executors which run out of process, as those can not
                           *currently* transfer stack frames across process
                           boundaries); if no callable is provided then a
                           default failure logging function will be used
                           instead (do note that
                           any user provided callable should not raise
                           exceptions on being called)
        :type on_failure: callable
        :param args: positional arguments to be passed to all callables
        :type args: tuple
        :param kwargs: keyword arguments to be passed to all callables
        :type kwargs: dict
        """
        callables = []
        for obj in objects:
            for name, member in inspect.getmembers(obj):
                if name.startswith('_') and exclude_hidden:
                    continue
                if callable(member):
                    missing_attrs = _check_attrs(member)
                    if not missing_attrs:
                        callables.append((member, args, kwargs))
        return cls(callables, log=log, executor_factory=executor_factory, cond_cls=cond_cls, event_cls=event_cls, schedule_strategy=schedule_strategy, now_func=now_func, on_failure=on_failure)

    def __init__(self, callables, log=None, executor_factory=None, cond_cls=threading.Condition, event_cls=threading.Event, schedule_strategy='last_started', now_func=utils.now, on_failure=None):
        """Creates a new worker using the given periodic callables.

        :param callables: a iterable of tuple objects previously decorated
                          with the :py:func:`.periodic` decorator, each item
                          in the iterable is expected to be in the format
                          of ``(cb, args, kwargs)`` where ``cb`` is the
                          decorated function and ``args`` and ``kwargs`` are
                          any positional and keyword arguments to send into
                          the callback when it is activated (both ``args``
                          and ``kwargs`` may be provided as none to avoid
                          using them)
        :type callables: iterable
        :param log: logger to use when creating a new worker (defaults
                    to the module logger if none provided), it is currently
                    only used to report callback failures (if they occur)
        :type log: logger
        :param executor_factory: factory callable that can be used to generate
                                 executor objects that will be used to
                                 run the periodic callables (if none is
                                 provided one will be created that uses
                                 the :py:class:`~futurist.SynchronousExecutor`
                                 class)
        :type executor_factory: ExecutorFactory or any callable
        :param cond_cls: callable object that can
                          produce ``threading.Condition``
                          (or compatible/equivalent) objects
        :type cond_cls: callable
        :param event_cls: callable object that can produce ``threading.Event``
                          (or compatible/equivalent) objects
        :type event_cls: callable
        :param schedule_strategy: string to select one of the built-in
                                  strategies that can return the
                                  next time a callable should run
        :type schedule_strategy: string
        :param now_func: callable that can return the current time offset
                         from some point (used in calculating elapsed times
                         and next times to run); preferably this is
                         monotonically increasing
        :type now_func: callable
        :param on_failure: callable that will be called whenever a periodic
                           function fails with an error, it will be provided
                           four positional arguments and one keyword
                           argument, the first positional argument being the
                           callable that failed, the second being the type
                           of activity under which it failed (``IMMEDIATE`` or
                           ``PERIODIC``), the third being the spacing that the
                           callable runs at and the fourth ``exc_info`` tuple
                           of the failure. The keyword argument ``traceback``
                           will also be provided that may be be a string
                           that caused the failure (this is required for
                           executors which run out of process, as those can not
                           *currently* transfer stack frames across process
                           boundaries); if no callable is provided then a
                           default failure logging function will be used
                           instead (do note that
                           any user provided callable should not raise
                           exceptions on being called)
        :type on_failure: callable
        """
        if on_failure is not None and (not callable(on_failure)):
            raise ValueError('On failure callback %r must be callable' % on_failure)
        self._tombstone = event_cls()
        self._waiter = cond_cls()
        self._dead = event_cls()
        self._active = event_cls()
        self._cond_cls = cond_cls
        self._watchers = []
        self._works = []
        for cb, args, kwargs in callables:
            if not callable(cb):
                raise ValueError('Periodic callback %r must be callable' % cb)
            missing_attrs = _check_attrs(cb)
            if missing_attrs:
                raise ValueError('Periodic callback %r missing required attributes %s' % (cb, missing_attrs))
            if cb._is_periodic:
                if args is None:
                    args = self._NO_OP_ARGS
                if kwargs is None:
                    kwargs = self._NO_OP_KWARGS.copy()
                cb_metrics = self._INITIAL_METRICS.copy()
                work = Work(utils.get_callback_name(cb), cb, args, kwargs)
                watcher = Watcher(cb_metrics, work)
                self._works.append(work)
                self._watchers.append((cb_metrics, watcher))
        try:
            strategy = self.BUILT_IN_STRATEGIES[schedule_strategy]
            self._schedule_strategy = strategy[0]
            self._initial_schedule_strategy = strategy[1]
        except KeyError:
            valid_strategies = sorted(self.BUILT_IN_STRATEGIES.keys())
            raise ValueError("Scheduling strategy '%s' must be one of %s selectable strategies" % (schedule_strategy, valid_strategies))
        self._immediates, self._schedule = _build(now_func, self._works, self._initial_schedule_strategy)
        self._log = log or LOG
        if executor_factory is None:
            executor_factory = lambda: futurist.SynchronousExecutor()
        if on_failure is None:
            on_failure = functools.partial(_on_failure_log, self._log)
        self._on_failure = on_failure
        self._executor_factory = executor_factory
        self._now_func = now_func

    def __len__(self):
        """How many callables/periodic work units are currently active."""
        return len(self._works)

    def _run(self, executor, runner, auto_stop_when_empty):
        """Main worker run loop."""
        barrier = utils.Barrier(cond_cls=self._cond_cls)
        rnd = random.SystemRandom()

        def _process_scheduled():
            with self._waiter:
                while not self._schedule and (not self._tombstone.is_set()) and (not self._immediates):
                    self._waiter.wait(self.MAX_LOOP_IDLE)
                if self._tombstone.is_set():
                    return
                if self._immediates:
                    return
                submitted_at = now = self._now_func()
                next_run, index = self._schedule.pop()
                when_next = next_run - now
                if when_next <= 0:
                    work = self._works[index]
                    self._log.debug("Submitting periodic callback '%s'", work.name)
                    try:
                        fut = executor.submit(runner.run, work)
                    except _SCHEDULE_RETRY_EXCEPTIONS as exc:
                        delay = self._RESCHEDULE_DELAY + rnd.random() * self._RESCHEDULE_JITTER
                        self._log.error("Failed to submit periodic callback '%s', retrying after %.2f sec. Error: %s", work.name, delay, exc)
                        self._schedule.push(self._now_func() + delay, index)
                    else:
                        barrier.incr()
                        fut.add_done_callback(functools.partial(_on_done, PERIODIC, work, index, submitted_at))
                else:
                    self._schedule.push(next_run, index)
                    when_next = min(when_next, self.MAX_LOOP_IDLE)
                    self._waiter.wait(when_next)

        def _process_immediates():
            with self._waiter:
                try:
                    index = self._immediates.popleft()
                except IndexError:
                    pass
                else:
                    work = self._works[index]
                    submitted_at = self._now_func()
                    self._log.debug("Submitting immediate callback '%s'", work.name)
                    try:
                        fut = executor.submit(runner.run, work)
                    except _SCHEDULE_RETRY_EXCEPTIONS as exc:
                        self._log.error("Failed to submit immediate callback '%s', retrying. Error: %s", work.name, exc)
                        self._immediates.append(index)
                    else:
                        barrier.incr()
                        fut.add_done_callback(functools.partial(_on_done, IMMEDIATE, work, index, submitted_at))

        def _on_done(kind, work, index, submitted_at, fut):
            cb = work.callback
            started_at, finished_at, failure = fut.result()
            cb_metrics, _watcher = self._watchers[index]
            cb_metrics['runs'] += 1
            schedule_again = True
            if failure is not None:
                if not issubclass(failure.exc_type, NeverAgain):
                    cb_metrics['failures'] += 1
                    try:
                        self._on_failure(cb, kind, cb._periodic_spacing, failure.exc_info, traceback=failure.traceback)
                    except Exception as exc:
                        self._log.error('On failure callback %r raised an unhandled exception. Error: %s', self._on_failure, exc)
                else:
                    cb_metrics['successes'] += 1
                    schedule_again = False
                    self._log.debug("Periodic callback '%s' raised 'NeverAgain' exception, stopping any further execution of it.", work.name)
            else:
                cb_metrics['successes'] += 1
            elapsed = max(0, finished_at - started_at)
            elapsed_waiting = max(0, started_at - submitted_at)
            cb_metrics['elapsed'] += elapsed
            cb_metrics['elapsed_waiting'] += elapsed_waiting
            with self._waiter:
                with barrier.decr_cm() as am_left:
                    if schedule_again:
                        next_run = self._schedule_strategy(cb, started_at, finished_at, cb_metrics)
                        self._schedule.push(next_run, index)
                    else:
                        cb_metrics['requested_stop'] = True
                        if am_left <= 0 and len(self._immediates) == 0 and (len(self._schedule) == 0) and auto_stop_when_empty:
                            self._tombstone.set()
                self._waiter.notify_all()
        try:
            while not self._tombstone.is_set():
                _process_immediates()
                _process_scheduled()
        finally:
            barrier.wait()

    def _on_finish(self):
        if not self._log.isEnabledFor(logging.DEBUG):
            return
        cols = list(_DEFAULT_COLS)
        for c in ['Runs in', 'Active', 'Periodicity']:
            cols.remove(c)
        self._log.debug('Stopped running %s callbacks:\n%s', len(self._works), self.pformat(columns=cols) if prettytable else 'statistics not available, PrettyTable missing')

    def pformat(self, columns=_DEFAULT_COLS):
        if prettytable is None:
            raise ImportError('PrettyTable is required to use the pformat method')
        if not isinstance(columns, (list, tuple)):
            columns = list(columns)
        if not columns:
            raise ValueError('At least one of %s columns must be provided' % set(_DEFAULT_COLS))
        for c in columns:
            if c not in _DEFAULT_COLS:
                raise ValueError("Unknown column '%s', valid column names are %s" % (c, set(_DEFAULT_COLS)))
        tbl_rows = []
        now = self._now_func()
        for index, work in enumerate(self._works):
            _cb_metrics, watcher = self._watchers[index]
            next_run = self._schedule.fetch_next_run(index)
            if watcher.requested_stop:
                active = False
                runs_in = 'n/a'
            elif next_run is None:
                active = True
                runs_in = 'n/a'
            else:
                active = False
                runs_in = '%0.4fs' % max(0.0, next_run - now)
            cb_row = {'Name': work.name, 'Active': active, 'Periodicity': work.callback._periodic_spacing, 'Runs': watcher.runs, 'Runs in': runs_in, 'Failures': watcher.failures, 'Successes': watcher.successes, 'Stop Requested': watcher.requested_stop}
            try:
                cb_row_avgs = ['%0.4fs' % watcher.average_elapsed, '%0.4fs' % watcher.average_elapsed_waiting]
            except ZeroDivisionError:
                cb_row_avgs = ['.', '.']
            cb_row['Average elapsed'] = cb_row_avgs[0]
            cb_row['Average elapsed waiting'] = cb_row_avgs[1]
            tbl_rows.append(cb_row)
        tbl = prettytable.PrettyTable(columns)
        for cb_row in tbl_rows:
            tbl_row = []
            for c in columns:
                tbl_row.append(cb_row[c])
            tbl.add_row(tbl_row)
        return tbl.get_string()

    def add(self, cb, *args, **kwargs):
        """Adds a new periodic callback to the current worker.

        Returns a :py:class:`.Watcher` if added successfully or the value
        ``None`` if not (or raises a ``ValueError`` if the callback is not
        correctly formed and/or decorated).

        :param cb: a callable object/method/function previously decorated
                   with the :py:func:`.periodic` decorator
        :type cb: callable
        """
        if not callable(cb):
            raise ValueError('Periodic callback %r must be callable' % cb)
        missing_attrs = _check_attrs(cb)
        if missing_attrs:
            raise ValueError('Periodic callback %r missing required attributes %s' % (cb, missing_attrs))
        if not cb._is_periodic:
            return None
        now = self._now_func()
        with self._waiter:
            cb_index = len(self._works)
            cb_metrics = self._INITIAL_METRICS.copy()
            work = Work(utils.get_callback_name(cb), cb, args, kwargs)
            watcher = Watcher(cb_metrics, work)
            self._works.append(work)
            self._watchers.append((cb_metrics, watcher))
            if cb._periodic_run_immediately:
                self._immediates.append(cb_index)
            else:
                next_run = self._initial_schedule_strategy(cb, now)
                self._schedule.push(next_run, cb_index)
            self._waiter.notify_all()
            return watcher

    def start(self, allow_empty=False, auto_stop_when_empty=False):
        """Starts running (will not return until :py:meth:`.stop` is called).

        :param allow_empty: instead of running with no callbacks raise when
                            this worker has no contained callables (this can be
                            set to true and :py:meth:`.add` can be used to add
                            new callables on demand), note that when enabled
                            and no callbacks exist this will block and
                            sleep (until either stopped or callbacks are
                            added)
        :type allow_empty: bool
        :param auto_stop_when_empty: when the provided periodic functions have
                                     all exited and this is false then the
                                     thread responsible for executing those
                                     methods will just spin/idle waiting for
                                     a new periodic function to be added;
                                     switching it to true will make this
                                     idling not happen (and instead when no
                                     more periodic work exists then the
                                     calling thread will just return).
        :type auto_stop_when_empty: bool
        """
        if not self._works and (not allow_empty):
            raise RuntimeError('A periodic worker can not start without any callables to process')
        if self._active.is_set():
            raise RuntimeError('A periodic worker can not be started twice')
        executor = self._executor_factory()
        if isinstance(executor, futures.ProcessPoolExecutor):
            runner = _Runner(self._now_func, retain_traceback=False)
        else:
            runner = _Runner(self._now_func, retain_traceback=True)
        self._dead.clear()
        self._active.set()
        try:
            self._run(executor, runner, auto_stop_when_empty)
        finally:
            if getattr(self._executor_factory, 'shutdown', True):
                executor.shutdown()
            self._dead.set()
            self._active.clear()
            self._on_finish()

    def stop(self):
        """Sets the tombstone (this stops any further executions)."""
        with self._waiter:
            self._tombstone.set()
            self._waiter.notify_all()

    def iter_watchers(self):
        """Iterator/generator over all the currently maintained watchers."""
        for _cb_metrics, watcher in self._watchers:
            yield watcher

    def reset(self):
        """Resets the workers internal state."""
        self._tombstone.clear()
        self._dead.clear()
        for cb_metrics, _watcher in self._watchers:
            for k in list(cb_metrics):
                cb_metrics[k] = 0
        self._immediates, self._schedule = _build(self._now_func, self._works, self._initial_schedule_strategy)

    def wait(self, timeout=None):
        """Waits for the :py:meth:`.start` method to gracefully exit.

        An optional timeout can be provided, which will cause the method to
        return within the specified timeout. If the timeout is reached, the
        returned value will be False.

        :param timeout: Maximum number of seconds that the :meth:`.wait`
                        method should block for
        :type timeout: float/int
        """
        self._dead.wait(timeout)
        return self._dead.is_set()