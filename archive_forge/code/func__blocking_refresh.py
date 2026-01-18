import abc
from enum import Enum
import os
from google.auth import _helpers, environment_vars
from google.auth import exceptions
from google.auth import metrics
from google.auth._refresh_worker import RefreshThreadManager
def _blocking_refresh(self, request):
    if not self.valid:
        self.refresh(request)