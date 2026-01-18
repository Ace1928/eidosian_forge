import os
import sys
import warnings
from datetime import datetime, timezone
from importlib import import_module
from typing import IO, TYPE_CHECKING, Any, List, Optional, cast
from kombu.utils.imports import symbol_by_name
from kombu.utils.objects import cached_property
from celery import _state, signals
from celery.exceptions import FixupWarning, ImproperlyConfigured
def _verify_django_version(django: 'ModuleType') -> None:
    if django.VERSION < (1, 11):
        raise ImproperlyConfigured('Celery 5.x requires Django 1.11 or later.')