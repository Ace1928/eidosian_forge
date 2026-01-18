import datetime
import logging
import os
import shutil
import tempfile
from django.conf import settings
from django.contrib.sessions.backends.base import (
from django.contrib.sessions.exceptions import InvalidSessionKey
from django.core.exceptions import ImproperlyConfigured, SuspiciousOperation
def _expiry_date(self, session_data):
    """
        Return the expiry time of the file storing the session's content.
        """
    return session_data.get('_session_expiry') or self._last_modification() + datetime.timedelta(seconds=self.get_session_cookie_age())