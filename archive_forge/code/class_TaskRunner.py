import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
class TaskRunner(object):
    """Wrapper for a resumable task (co-routine)."""

    def __init__(self, task, *args, **kwargs):
        """Initialise with a task function and arguments.

        The arguments are passed to task when it is started.

        The task function may be a co-routine that yields control flow between
        steps.

        If the task co-routine wishes to be advanced only on every nth step of
        the TaskRunner, it may yield an integer which is the period of the
        task. e.g. "yield 2" will result in the task being advanced on every
        second step.
        """
        assert callable(task), 'Task is not callable'
        self._task = task
        self._args = args
        self._kwargs = kwargs
        self._runner = None
        self._done = False
        self._timeout = None
        self._poll_period = 1
        self.name = task_description(task)

    def __str__(self):
        """Return a human-readable string representation of the task."""
        text = 'Task %s' % self.name
        return str(text)

    def _sleep(self, wait_time):
        """Sleep for the specified number of seconds."""
        if ENABLE_SLEEP and wait_time is not None:
            LOG.debug('%s sleeping', str(self))
            eventlet.sleep(wait_time)

    def __call__(self, wait_time=1, timeout=None, progress_callback=None):
        """Start and run the task to completion.

        The task will first sleep for zero seconds, then sleep for `wait_time`
        seconds between steps. To avoid sleeping, pass `None` for `wait_time`.
        """
        assert self._runner is None, 'Task already started'
        started = False
        for step in self.as_task(timeout=timeout, progress_callback=progress_callback):
            self._sleep(wait_time if started or wait_time is None else 0)
            started = True

    def start(self, timeout=None):
        """Initialise the task and run its first step.

        If a timeout is specified, any attempt to step the task after that
        number of seconds has elapsed will result in a Timeout being
        raised inside the task.
        """
        assert self._runner is None, 'Task already started'
        assert not self._done, 'Task already cancelled'
        LOG.debug('%s starting', str(self))
        if timeout is not None:
            self._timeout = Timeout(self, timeout)
        result = self._task(*self._args, **self._kwargs)
        if isinstance(result, types.GeneratorType):
            self._runner = result
            self.step()
        else:
            self._runner = False
            self._done = True
            LOG.debug('%s done (not resumable)', str(self))

    def step(self):
        """Run another step of the task.

        Return True if the task is complete; False otherwise.
        """
        if not self.done():
            assert self._runner is not None, 'Task not started'
            if self._poll_period > 1:
                self._poll_period -= 1
                return False
            if self._timeout is not None and self._timeout.expired():
                LOG.info('%s timed out', self)
                self._done = True
                self._timeout.trigger(self._runner)
            else:
                LOG.debug('%s running', str(self))
                try:
                    poll_period = next(self._runner)
                except StopIteration:
                    self._done = True
                    LOG.debug('%s complete', str(self))
                else:
                    if isinstance(poll_period, int):
                        self._poll_period = max(poll_period, 1)
                    else:
                        self._poll_period = 1
        return self._done

    def run_to_completion(self, wait_time=1, progress_callback=None):
        """Run the task to completion.

        The task will sleep for `wait_time` seconds between steps. To avoid
        sleeping, pass `None` for `wait_time`.
        """
        assert self._runner is not None, 'Task not started'
        for step in self.as_task(progress_callback=progress_callback):
            self._sleep(wait_time)

    def as_task(self, timeout=None, progress_callback=None):
        """Return a task that drives the TaskRunner."""
        resuming = self.started()
        if not resuming:
            self.start(timeout=timeout)
        elif timeout is not None:
            new_timeout = Timeout(self, timeout)
            if new_timeout.earlier_than(self._timeout):
                self._timeout = new_timeout
        done = self.step() if resuming else self.done()
        while not done:
            try:
                yield
                if progress_callback is not None:
                    progress_callback()
            except GeneratorExit:
                self.cancel()
                raise
            except:
                self._done = True
                try:
                    self._runner.throw(*sys.exc_info())
                except StopIteration:
                    return
                else:
                    self._done = False
            else:
                done = self.step()

    def cancel(self, grace_period=None):
        """Cancel the task and mark it as done."""
        if self.done():
            return
        if not self.started() or grace_period is None:
            LOG.debug('%s cancelled', str(self))
            self._done = True
            if self.started():
                self._runner.close()
        else:
            timeout = TimedCancel(self, grace_period)
            if timeout.earlier_than(self._timeout):
                self._timeout = timeout

    def started(self):
        """Return True if the task has been started."""
        return self._runner is not None

    def done(self):
        """Return True if the task is complete."""
        return self._done

    def __nonzero__(self):
        """Return True if there are steps remaining."""
        return not self.done()

    def __bool__(self):
        """Return True if there are steps remaining."""
        return self.__nonzero__()