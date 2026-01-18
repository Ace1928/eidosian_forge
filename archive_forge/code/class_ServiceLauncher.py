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
class ServiceLauncher(Launcher):
    """Runs one or more service in a parent process."""

    def __init__(self, conf, restart_method='reload'):
        """Constructor.

        :param conf: an instance of ConfigOpts
        :param restart_method: passed to super
        """
        super(ServiceLauncher, self).__init__(conf, restart_method=restart_method)
        self.signal_handler = SignalHandler()

    def _graceful_shutdown(self, *args):
        self.signal_handler.clear()
        if self.conf.graceful_shutdown_timeout and self.signal_handler.is_signal_supported('SIGALRM'):
            signal.alarm(self.conf.graceful_shutdown_timeout)
        self.stop()

    def _reload_service(self, *args):
        self.signal_handler.clear()
        raise SignalExit(signal.SIGHUP)

    def _fast_exit(self, *args):
        LOG.info('Caught SIGINT signal, instantaneous exiting')
        os._exit(1)

    def _on_timeout_exit(self, *args):
        LOG.info('Graceful shutdown timeout exceeded, instantaneous exiting')
        os._exit(1)

    def handle_signal(self):
        """Set self._handle_signal as a signal handler."""
        self.signal_handler.clear()
        self.signal_handler.add_handler('SIGTERM', self._graceful_shutdown)
        self.signal_handler.add_handler('SIGINT', self._fast_exit)
        self.signal_handler.add_handler('SIGHUP', self._reload_service)
        self.signal_handler.add_handler('SIGALRM', self._on_timeout_exit)

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

    def wait(self):
        """Wait for a service to terminate and restart it on SIGHUP.

        :returns: termination status
        """
        systemd.notify_once()
        self.signal_handler.clear()
        while True:
            self.handle_signal()
            status, signo = self._wait_for_exit_or_signal()
            if not _is_sighup_and_daemon(signo):
                break
            self.restart()
        super(ServiceLauncher, self).wait()
        return status