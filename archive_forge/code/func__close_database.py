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
def _close_database(self, force: bool=False) -> None:
    for conn in self._db.connections.all():
        try:
            if force:
                conn.close()
            else:
                conn.close_if_unusable_or_obsolete()
        except self.interface_errors:
            pass
        except self.DatabaseError as exc:
            str_exc = str(exc)
            if 'closed' not in str_exc and 'not connected' not in str_exc:
                raise