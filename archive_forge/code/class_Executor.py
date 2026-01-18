from tensorflow.python import pywrap_tfe
class Executor(object):
    """A class for handling eager execution.

  The default behavior for asynchronous execution is to serialize all ops on
  a single thread. Having different `Executor` objects in different threads
  enables executing ops asynchronously in parallel:

  ```python
  def thread_function():
    executor = executor.Executor(enable_async=True):
    context.set_executor(executor)

  a = threading.Thread(target=thread_function)
  a.start()
  b = threading.Thread(target=thread_function)
  b.start()
  ```
  """
    __slots__ = ['_handle']

    def __init__(self, handle):
        self._handle = handle

    def __del__(self):
        try:
            self.wait()
            pywrap_tfe.TFE_DeleteExecutor(self._handle)
        except TypeError:
            pass

    def is_async(self):
        return pywrap_tfe.TFE_ExecutorIsAsync(self._handle)

    def handle(self):
        return self._handle

    def wait(self):
        """Waits for ops dispatched in this executor to finish."""
        pywrap_tfe.TFE_ExecutorWaitForAllPendingNodes(self._handle)

    def clear_error(self):
        """Clears errors raised in this executor during execution."""
        pywrap_tfe.TFE_ExecutorClearError(self._handle)