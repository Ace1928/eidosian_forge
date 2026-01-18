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
def _wait_for_exit_or_signal(self):
    status = None
    signo = 0
    if self.conf.log_options:
        LOG.debug('Full set of CONF:')
        self.conf.log_opt_values(LOG, logging.DEBUG)
    try:
        super(ServiceLauncher, self).wait()
    except SignalExit as exc:
        signame = self.signal_handler.signals_to_name[exc.signo]
        LOG.info('Caught %s, handling', signame)
        status = exc.code
        signo = exc.signo
    except SystemExit as exc:
        self.stop()
        status = exc.code
    except Exception:
        self.stop()
    return (status, signo)