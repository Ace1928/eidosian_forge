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
def _show_deprecation_warning(self, message, category):
    stack = traceback.extract_stack()
    filename, _, _, _ = stack[-4]
    if not filename.startswith(os.path.dirname(django.__file__)):
        warnings.warn(message, category, stacklevel=2)