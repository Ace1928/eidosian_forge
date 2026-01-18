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
def _maybe_close_fd(fh: IO) -> None:
    try:
        os.close(fh.fileno())
    except (AttributeError, OSError, TypeError):
        pass