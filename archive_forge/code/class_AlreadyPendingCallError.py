import warnings
class AlreadyPendingCallError(Exception):
    """Raised when `reset`, or `step` is called asynchronously (e.g. with `reset_async`, or `step_async` respectively), and `reset_async`, or `step_async` (respectively) is called again (without a complete call to `reset_wait`, or `step_wait` respectively)."""

    def __init__(self, message: str, name: str):
        """Initialises the exception with name attributes."""
        super().__init__(message)
        self.name = name