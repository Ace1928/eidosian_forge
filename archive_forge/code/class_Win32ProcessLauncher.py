import abc
import errno
import functools
import os
import re
import signal
import struct
import subprocess
import sys
import time
from eventlet.green import socket
import eventlet.greenio
import eventlet.wsgi
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from osprofiler import opts as profiler_opts
import routes.middleware
import webob.dec
import webob.exc
from webob import multidict
from glance.common import config
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
import glance.db
from glance import housekeeping
from glance import i18n
from glance.i18n import _, _LE, _LI, _LW
from glance import sqlite_migration
class Win32ProcessLauncher(object):

    def __init__(self):
        self._processutils = os_win_utilsfactory.get_processutils()
        self._workers = []
        self._worker_job_handles = []

    def add_process(self, cmd):
        LOG.info('Starting subprocess: %s', cmd)
        worker = subprocess.Popen(cmd, close_fds=False)
        try:
            job_handle = self._processutils.kill_process_on_job_close(worker.pid)
        except Exception:
            LOG.exception('Could not associate child process with a job, killing it.')
            worker.kill()
            raise
        self._worker_job_handles.append(job_handle)
        self._workers.append(worker)
        return worker

    def wait(self):
        pids = [worker.pid for worker in self._workers]
        if pids:
            self._processutils.wait_for_multiple_processes(pids, wait_all=True)
        time.sleep(0)