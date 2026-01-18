import threading
from tensorboard import errors
def FirstEventTimestamp(self, run):
    """Return the timestamp of the first DebugEvent of the given run.

        This may perform I/O if no events have been loaded yet for the run.

        Args:
          run: A string name of the run for which the timestamp is retrieved.
            This currently must be hardcoded as `DEFAULT_DEBUGGER_RUN_NAME`,
            as each logdir contains at most one DebugEvent file set (i.e., a
            run of a tfdbg2-instrumented TensorFlow program.)

        Returns:
            The wall_time of the first event of the run, which will be in seconds
            since the epoch as a `float`.
        """
    if self._reader is None:
        raise ValueError('No tfdbg2 runs exists.')
    if run != DEFAULT_DEBUGGER_RUN_NAME:
        raise ValueError('Expected run name to be %s, but got %s' % (DEFAULT_DEBUGGER_RUN_NAME, run))
    return self._reader.starting_wall_time()