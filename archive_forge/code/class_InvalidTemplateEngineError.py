import functools
from collections import Counter
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class InvalidTemplateEngineError(ImproperlyConfigured):
    pass