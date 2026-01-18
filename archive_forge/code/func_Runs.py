import threading
from tensorboard import errors
def Runs(self):
    """Return all the tfdbg2 run names in the logdir watched by this instance.

        The `Run()` method of this class is specialized for the tfdbg2-format
        DebugEvent files.

        As a side effect, this method unblocks the underlying reader's period
        reloading if a reader exists. This lets the reader update at a higher
        frequency than the default one with 30-second sleeping period between
        reloading when data is being queried actively from this instance.
        Note that this `Runs()` method is used by all other public data-access
        methods of this class (e.g., `ExecutionData()`, `GraphExecutionData()`).
        Hence calls to those methods will lead to accelerated data reloading of
        the reader.

        Returns:
          If tfdbg2-format data exists in the `logdir` of this object, returns:
              ```
              {runName: { "debugger-v2": [tag1, tag2, tag3] } }
              ```
              where `runName` is the hard-coded string `DEFAULT_DEBUGGER_RUN_NAME`
              string. This is related to the fact that tfdbg2 currently contains
              at most one DebugEvent file set per directory.
          If no tfdbg2-format data exists in the `logdir`, an empty `dict`.
        """
    self._tryCreateReader()
    if self._reader:
        self._reloadReader()
        return {DEFAULT_DEBUGGER_RUN_NAME: {'debugger-v2': []}}
    else:
        return {}