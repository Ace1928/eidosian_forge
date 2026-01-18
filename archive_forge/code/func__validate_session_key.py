import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def _validate_session_key(self, key):
    """
        Key must be truthy and at least 8 characters long. 8 characters is an
        arbitrary lower bound for some minimal key security.
        """
    return key and len(key) >= 8