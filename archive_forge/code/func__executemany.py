import datetime
import decimal
import functools
import logging
import time
import warnings
from contextlib import contextmanager
from hashlib import md5
from django.apps import apps
from django.db import NotSupportedError
from django.utils.dateparse import parse_time
def _executemany(self, sql, param_list, *ignored_wrapper_args):
    if not apps.ready and (not apps.stored_app_configs):
        warnings.warn(self.APPS_NOT_READY_WARNING_MSG, category=RuntimeWarning)
    self.db.validate_no_broken_transaction()
    with self.db.wrap_database_errors:
        return self.cursor.executemany(sql, param_list)