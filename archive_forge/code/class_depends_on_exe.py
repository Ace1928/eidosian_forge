import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
class depends_on_exe(object):
    """Decorator to skip test if an executable is unavailable"""

    def __init__(self, exe):
        self.exe = exe

    def __call__(self, func):

        def _runner(*args, **kw):
            if os.name != 'nt':
                cmd = 'which %s' % self.exe
            else:
                cmd = ('where.exe', '%s' % self.exe)
            exitcode, out, err = execute(cmd, raise_error=False)
            if exitcode != 0:
                args[0].disabled_message = 'test requires exe: %s' % self.exe
                args[0].disabled = True
            func(*args, **kw)
        _runner.__name__ = func.__name__
        _runner.__doc__ = func.__doc__
        return _runner