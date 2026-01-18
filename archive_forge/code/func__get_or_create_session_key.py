import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def _get_or_create_session_key(self):
    if self._session_key is None:
        self._session_key = self._get_new_session_key()
    return self._session_key