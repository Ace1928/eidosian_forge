import datetime
import logging
import os
import shutil
import tempfile
from django.conf import settings
from django.contrib.sessions.backends.base import (
from django.contrib.sessions.exceptions import InvalidSessionKey
from django.core.exceptions import ImproperlyConfigured, SuspiciousOperation
def _last_modification(self):
    """
        Return the modification time of the file storing the session's content.
        """
    modification = os.stat(self._key_to_file()).st_mtime
    tz = datetime.timezone.utc if settings.USE_TZ else None
    return datetime.datetime.fromtimestamp(modification, tz=tz)