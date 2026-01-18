import warnings
class NoAsyncCallError(Exception):
    """Raised when an asynchronous `reset`, or `step` is not running, but `reset_wait`, or `step_wait` (respectively) is called."""

    def __init__(self, message: str, name: str):
        """Initialises the exception with name attributes."""
        super().__init__(message)
        self.name = name