import abc
from enum import Enum
import os
from google.auth import _helpers, environment_vars
from google.auth import exceptions
from google.auth import metrics
from google.auth._refresh_worker import RefreshThreadManager
def _non_blocking_refresh(self, request):
    use_blocking_refresh_fallback = False
    if self.token_state == TokenState.STALE:
        use_blocking_refresh_fallback = not self._refresh_worker.start_refresh(self, request)
    if self.token_state == TokenState.INVALID or use_blocking_refresh_fallback:
        self.refresh(request)
        self._refresh_worker.clear_error()