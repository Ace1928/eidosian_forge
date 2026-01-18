import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def _get_new_session_key(self):
    """Return session key that isn't being used."""
    while True:
        session_key = get_random_string(32, VALID_KEY_CHARS)
        if not self.exists(session_key):
            return session_key