from typing import ContextManager
from google.api_core.exceptions import FailedPrecondition
class RequireStarted(ContextManager):

    def __init__(self):
        self._started = False

    def __enter__(self):
        if self._started:
            raise FailedPrecondition('__enter__ called twice.')
        self._started = True
        return self

    def require_started(self):
        if not self._started:
            raise FailedPrecondition('__enter__ has never been called.')

    def __exit__(self, exc_type, exc_value, traceback):
        self.require_started()