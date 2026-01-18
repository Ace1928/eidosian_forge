import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
class JitteredRetry(Retry):
    """
    urllib3 < 2 doesn't support `backoff_jitter`. This class is a workaround for that.
    """

    def __init__(self, *args, backoff_jitter=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.backoff_jitter = backoff_jitter

    def get_backoff_time(self):
        """
        Source: https://github.com/urllib3/urllib3/commit/214b184923388328919b0a4b0c15bff603aa51be
        """
        backoff_value = super().get_backoff_time()
        if self.backoff_jitter != 0.0:
            backoff_value += random.random() * self.backoff_jitter
        default_backoff = Retry.BACKOFF_MAX if Version(urllib3.__version__) < Version('1.26.9') else Retry.DEFAULT_BACKOFF_MAX
        return float(max(0, min(default_backoff, backoff_value)))