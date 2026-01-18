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
class ProcessLauncher(object):
    """Launch a service with a given number of workers."""

    def __init__(self, conf, wait_interval=0.01, restart_method='reload'):
        """Constructor.

        :param conf: an instance of ConfigOpts
        :param wait_interval: The interval to sleep for between checks
                              of child process exit.
        :param restart_method: If 'reload', calls reload_config_files on
            SIGHUP. If 'mutate', calls mutate_config_files on SIGHUP. Other
            values produce a ValueError.
        """
        self.conf = conf
        conf.register_opts(_options.service_opts)
        self.children = {}
        self.sigcaught = None
        self.running = True
        self.wait_interval = wait_interval
        self.launcher = None
        rfd, self.writepipe = os.pipe()
        self.readpipe = eventlet.greenio.GreenPipe(rfd, 'r')
        self.signal_handler = SignalHandler()
        self.handle_signal()
        self.restart_method = restart_method
        if restart_method not in _LAUNCHER_RESTART_METHODS:
            raise ValueError(_('Invalid restart_method: %s') % restart_method)

    def handle_signal(self):
        """Add instance's signal handlers to class handlers."""
        self.signal_handler.add_handler('SIGTERM', self._handle_term)
        self.signal_handler.add_handler('SIGHUP', self._handle_hup)
        self.signal_handler.add_handler('SIGINT', self._fast_exit)
        self.signal_handler.add_handler('SIGALRM', self._on_alarm_exit)

    def _handle_term(self, signo, frame):
        """Handle a TERM event.

        :param signo: signal number
        :param frame: current stack frame
        """
        self.sigcaught = signo
        self.running = False
        self.signal_handler.clear()

    def _handle_hup(self, signo, frame):
        """Handle a HUP event.

        :param signo: signal number
        :param frame: current stack frame
        """
        self.sigcaught = signo
        self.running = False

    def _fast_exit(self, signo, frame):
        LOG.info('Caught SIGINT signal, instantaneous exiting')
        os._exit(1)

    def _on_alarm_exit(self, signo, frame):
        LOG.info('Graceful shutdown timeout exceeded, instantaneous exiting')
        os._exit(1)

    def _pipe_watcher(self):
        self.readpipe.read(1)
        LOG.info('Parent process has died unexpectedly, exiting')
        if self.launcher:
            self.launcher.stop()
        sys.exit(1)

    def _child_process_handle_signal(self):

        def _sigterm(*args):
            self.signal_handler.clear()
            self.launcher.stop()

        def _sighup(*args):
            self.signal_handler.clear()
            raise SignalExit(signal.SIGHUP)
        self.signal_handler.clear()
        self.signal_handler.add_handler('SIGTERM', _sigterm)
        self.signal_handler.add_handler('SIGHUP', _sighup)
        self.signal_handler.add_handler('SIGINT', self._fast_exit)

    def _child_wait_for_exit_or_signal(self, launcher):
        status = 0
        signo = 0
        try:
            launcher.wait()
        except SignalExit as exc:
            signame = self.signal_handler.signals_to_name[exc.signo]
            LOG.info('Child caught %s, handling', signame)
            status = exc.code
            signo = exc.signo
        except SystemExit as exc:
            launcher.stop()
            status = exc.code
        except BaseException:
            launcher.stop()
            LOG.exception('Unhandled exception')
            status = 2
        return (status, signo)

    def _child_process(self, service):
        self._child_process_handle_signal()
        eventlet.hubs.use_hub()
        os.close(self.writepipe)
        eventlet.spawn_n(self._pipe_watcher)
        random.seed()
        launcher = Launcher(self.conf, restart_method=self.restart_method)
        launcher.launch_service(service)
        return launcher

    def _start_child(self, wrap):
        if len(wrap.forktimes) > wrap.workers:
            if time.time() - wrap.forktimes[0] < wrap.workers:
                LOG.info('Forking too fast, sleeping')
                time.sleep(1)
            wrap.forktimes.pop(0)
        wrap.forktimes.append(time.time())
        pid = os.fork()
        if pid == 0:
            tpool.killall()
            self.launcher = self._child_process(wrap.service)
            while True:
                self._child_process_handle_signal()
                status, signo = self._child_wait_for_exit_or_signal(self.launcher)
                if not _is_sighup_and_daemon(signo):
                    self.launcher.wait()
                    break
                self.launcher.restart()
            os._exit(status)
        LOG.debug('Started child %d', pid)
        wrap.children.add(pid)
        self.children[pid] = wrap
        return pid

    def launch_service(self, service, workers=1):
        """Launch a service with a given number of workers.

       :param service: a service to launch, must be an instance of
              :class:`oslo_service.service.ServiceBase`
       :param workers: a number of processes in which a service
              will be running
        """
        _check_service_base(service)
        wrap = ServiceWrapper(service, workers)
        if hasattr(gc, 'freeze'):
            gc.freeze()
        LOG.info('Starting %d workers', wrap.workers)
        while self.running and len(wrap.children) < wrap.workers:
            self._start_child(wrap)

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

    def _respawn_children(self):
        while self.running:
            wrap = self._wait_child()
            if not wrap:
                eventlet.greenthread.sleep(self.wait_interval)
                continue
            while self.running and len(wrap.children) < wrap.workers:
                self._start_child(wrap)

    def wait(self):
        """Loop waiting on children to die and respawning as necessary."""
        systemd.notify_once()
        if self.conf.log_options:
            LOG.debug('Full set of CONF:')
            self.conf.log_opt_values(LOG, logging.DEBUG)
        try:
            while True:
                self.handle_signal()
                self._respawn_children()
                if not self.sigcaught:
                    return
                signame = self.signal_handler.signals_to_name[self.sigcaught]
                LOG.info('Caught %s, stopping children', signame)
                if not _is_sighup_and_daemon(self.sigcaught):
                    break
                child_signal = signal.SIGTERM
                if self.restart_method == 'reload':
                    self.conf.reload_config_files()
                elif self.restart_method == 'mutate':
                    self.conf.mutate_config_files()
                    child_signal = signal.SIGHUP
                for service in set([wrap.service for wrap in self.children.values()]):
                    service.reset()
                for pid in self.children:
                    os.kill(pid, child_signal)
                self.running = True
                self.sigcaught = None
        except eventlet.greenlet.GreenletExit:
            LOG.info('Wait called after thread killed. Cleaning up.')
        if self.conf.graceful_shutdown_timeout and self.signal_handler.is_signal_supported('SIGALRM'):
            signal.alarm(self.conf.graceful_shutdown_timeout)
        self.stop()

    def stop(self):
        """Terminate child processes and wait on each."""
        self.running = False
        LOG.debug('Stop services.')
        for service in set([wrap.service for wrap in self.children.values()]):
            service.stop()
        LOG.debug('Killing children.')
        for pid in self.children:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise
        if self.children:
            LOG.info('Waiting on %d children to exit', len(self.children))
            while self.children:
                self._wait_child()