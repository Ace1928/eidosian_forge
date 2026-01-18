import importlib
import os
import time
import traceback
import warnings
from pathlib import Path
import django
from django.conf import global_settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import RemovedInDjango51Warning, RemovedInDjango60Warning
from django.utils.functional import LazyObject, empty
@staticmethod
def _add_script_prefix(value):
    """
        Add SCRIPT_NAME prefix to relative paths.

        Useful when the app is being served at a subpath and manually prefixing
        subpath to STATIC_URL and MEDIA_URL in settings is inconvenient.
        """
    if value.startswith(('http://', 'https://', '/')):
        return value
    from django.urls import get_script_prefix
    return '%s%s' % (get_script_prefix(), value)