import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.LooperThread'])
class LooperThread(threading.Thread):
    """A thread that runs code repeatedly, optionally on a timer.

  This thread class is intended to be used with a `Coordinator`.  It repeatedly
  runs code specified either as `target` and `args` or by the `run_loop()`
  method.

  Before each run the thread checks if the coordinator has requested stop.  In
  that case the looper thread terminates immediately.

  If the code being run raises an exception, that exception is reported to the
  coordinator and the thread terminates.  The coordinator will then request all
  the other threads it coordinates to stop.

  You typically pass looper threads to the supervisor `Join()` method.
  """

    def __init__(self, coord, timer_interval_secs, target=None, args=None, kwargs=None):
        """Create a LooperThread.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Time boundaries at which to call Run(), or None
        if it should be called back to back.
      target: Optional callable object that will be executed in the thread.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
        if not isinstance(coord, Coordinator):
            raise ValueError("'coord' argument must be a Coordinator: %s" % coord)
        super(LooperThread, self).__init__()
        self.daemon = True
        self._coord = coord
        self._timer_interval_secs = timer_interval_secs
        self._target = target
        if self._target:
            self._args = args or ()
            self._kwargs = kwargs or {}
        elif args or kwargs:
            raise ValueError("'args' and 'kwargs' argument require that you also pass 'target'")
        self._coord.register_thread(self)

    @staticmethod
    def loop(coord, timer_interval_secs, target, args=None, kwargs=None):
        """Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(args)`
    repeatedly.  Otherwise `target(args)` is called every `timer_interval_secs`
    seconds.  The thread terminates when a stop of the coordinator is
    requested.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    """
        looper = LooperThread(coord, timer_interval_secs, target=target, args=args, kwargs=kwargs)
        looper.start()
        return looper

    def run(self):
        with self._coord.stop_on_exception():
            self.start_loop()
            if self._timer_interval_secs is None:
                while not self._coord.should_stop():
                    self.run_loop()
            else:
                next_timer_time = time.time()
                while not self._coord.wait_for_stop(next_timer_time - time.time()):
                    next_timer_time += self._timer_interval_secs
                    self.run_loop()
            self.stop_loop()

    def start_loop(self):
        """Called when the thread starts."""
        pass

    def stop_loop(self):
        """Called when the thread stops."""
        pass

    def run_loop(self):
        """Called at 'timer_interval_secs' boundaries."""
        if self._target:
            self._target(*self._args, **self._kwargs)