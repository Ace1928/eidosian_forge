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
def _execute_with_wrappers(self, sql, params, many, executor):
    context = {'connection': self.db, 'cursor': self}
    for wrapper in reversed(self.db.execute_wrappers):
        executor = functools.partial(wrapper, executor)
    return executor(sql, params, many, context)