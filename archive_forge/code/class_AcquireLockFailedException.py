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
class AcquireLockFailedException(Exception):

    def __init__(self, lock_name):
        self.message = 'Failed to acquire the lock %s' % lock_name

    def __str__(self):
        return self.message