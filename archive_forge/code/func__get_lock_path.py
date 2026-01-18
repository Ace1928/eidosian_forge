import contextlib
import errno
import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import fasteners
from oslo_config import cfg
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def _get_lock_path(name, lock_file_prefix, lock_path=None):
    name = name.replace(os.sep, '_')
    if lock_file_prefix:
        sep = '' if lock_file_prefix.endswith('-') else '-'
        name = '%s%s%s' % (lock_file_prefix, sep, name)
    local_lock_path = lock_path or CONF.oslo_concurrency.lock_path
    if not local_lock_path:
        raise cfg.RequiredOptError('lock_path', 'oslo_concurrency')
    return os.path.join(local_lock_path, name)