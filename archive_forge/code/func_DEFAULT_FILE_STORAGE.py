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
@property
def DEFAULT_FILE_STORAGE(self):
    self._show_deprecation_warning(DEFAULT_FILE_STORAGE_DEPRECATED_MSG, RemovedInDjango51Warning)
    return self.__getattr__('DEFAULT_FILE_STORAGE')