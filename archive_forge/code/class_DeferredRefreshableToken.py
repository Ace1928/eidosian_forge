import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import NamedTuple, Optional
import dateutil.parser
from dateutil.tz import tzutc
from botocore import UNSIGNED
from botocore.compat import total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.utils import CachedProperty, JSONFileCache, SSOTokenLoader
class DeferredRefreshableToken:
    _advisory_refresh_timeout = 15 * 60
    _mandatory_refresh_timeout = 10 * 60
    _attempt_timeout = 60

    def __init__(self, method, refresh_using, time_fetcher=_utc_now):
        self._time_fetcher = time_fetcher
        self._refresh_using = refresh_using
        self.method = method
        self._refresh_lock = threading.Lock()
        self._frozen_token = None
        self._next_refresh = None

    def get_frozen_token(self):
        self._refresh()
        return self._frozen_token

    def _refresh(self):
        refresh_type = self._should_refresh()
        if not refresh_type:
            return None
        block_for_refresh = refresh_type == 'mandatory'
        if self._refresh_lock.acquire(block_for_refresh):
            try:
                self._protected_refresh()
            finally:
                self._refresh_lock.release()

    def _protected_refresh(self):
        refresh_type = self._should_refresh()
        if not refresh_type:
            return None
        try:
            now = self._time_fetcher()
            self._next_refresh = now + timedelta(seconds=self._attempt_timeout)
            self._frozen_token = self._refresh_using()
        except Exception:
            logger.warning('Refreshing token failed during the %s refresh period.', refresh_type, exc_info=True)
            if refresh_type == 'mandatory':
                raise
        if self._is_expired():
            raise TokenRetrievalError(provider=self.method, error_msg='Token has expired and refresh failed')

    def _is_expired(self):
        if self._frozen_token is None:
            return False
        expiration = self._frozen_token.expiration
        remaining = total_seconds(expiration - self._time_fetcher())
        return remaining <= 0

    def _should_refresh(self):
        if self._frozen_token is None:
            return 'mandatory'
        expiration = self._frozen_token.expiration
        if expiration is None:
            return None
        now = self._time_fetcher()
        if now < self._next_refresh:
            return None
        remaining = total_seconds(expiration - now)
        if remaining < self._mandatory_refresh_timeout:
            return 'mandatory'
        elif remaining < self._advisory_refresh_timeout:
            return 'advisory'
        return None