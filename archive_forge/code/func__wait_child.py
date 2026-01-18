import abc
import collections
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import os
import random
import signal
import sys
import time
import eventlet
from eventlet import event
from eventlet import tpool
from oslo_concurrency import lockutils
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import eventlet_backdoor
from oslo_service import systemd
from oslo_service import threadgroup
def _wait_child(self):
    try:
        pid, status = os.waitpid(0, os.WNOHANG)
        if not pid:
            return None
    except OSError as exc:
        if exc.errno not in (errno.EINTR, errno.ECHILD):
            raise
        return None
    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        LOG.info('Child %(pid)d killed by signal %(sig)d', dict(pid=pid, sig=sig))
    else:
        code = os.WEXITSTATUS(status)
        LOG.info('Child %(pid)s exited with status %(code)d', dict(pid=pid, code=code))
    if pid not in self.children:
        LOG.warning('pid %d not in child list', pid)
        return None
    wrap = self.children.pop(pid)
    wrap.children.remove(pid)
    return wrap