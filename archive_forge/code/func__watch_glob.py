import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter
import django
from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
def _watch_glob(self, directory, patterns):
    """
        Watch a directory with a specific glob. If the directory doesn't yet
        exist, attempt to watch the parent directory and amend the patterns to
        include this. It's important this method isn't called more than one per
        directory when updating all subscriptions. Subsequent calls will
        overwrite the named subscription, so it must include all possible glob
        expressions.
        """
    prefix = 'glob'
    if not directory.exists():
        if not directory.parent.exists():
            logger.warning('Unable to watch directory %s as neither it or its parent exist.', directory)
            return
        prefix = 'glob-parent-%s' % directory.name
        patterns = ['%s/%s' % (directory.name, pattern) for pattern in patterns]
        directory = directory.parent
    expression = ['anyof']
    for pattern in patterns:
        expression.append(['match', pattern, 'wholename'])
    self._subscribe(directory, '%s:%s' % (prefix, directory), expression)