import copy
import enum
import functools
import logging
import multiprocessing
import shlex
import sys
import threading
from oslo_config import cfg
from oslo_config import types
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import daemon
class PrivContext(object):

    def __init__(self, prefix, cfg_section='privsep', pypath=None, capabilities=None, logger_name='oslo_privsep.daemon', timeout=None):
        if capabilities is None:
            raise ValueError('capabilities is a required parameter')
        self.pypath = pypath
        self.prefix = prefix
        self.cfg_section = cfg_section
        self.client_mode = sys.platform != 'win32'
        self.channel = None
        self.start_lock = threading.Lock()
        cfg.CONF.register_opts(OPTS, group=cfg_section)
        cfg.CONF.set_default('capabilities', group=cfg_section, default=capabilities)
        cfg.CONF.set_default('logger_name', group=cfg_section, default=logger_name)
        self.timeout = timeout

    @property
    def conf(self):
        """Return the oslo.config section object as lazily as possible."""
        return cfg.CONF[self.cfg_section]

    def __repr__(self):
        return 'PrivContext(cfg_section=%s)' % self.cfg_section

    def helper_command(self, sockpath):
        if self.pypath is None:
            raise AssertionError('helper_command requires priv_context pypath to be specified')
        if importutils.import_class(self.pypath) is not self:
            raise AssertionError('helper_command requires priv_context pypath for context object')
        if self.conf.helper_command:
            cmd = shlex.split(self.conf.helper_command)
        else:
            cmd = _HELPER_COMMAND_PREFIX + ['privsep-helper']
            try:
                for cfg_file in cfg.CONF.config_file:
                    cmd.extend(['--config-file', cfg_file])
            except cfg.NoSuchOptError:
                pass
            try:
                if cfg.CONF.config_dir is not None:
                    for cfg_dir in cfg.CONF.config_dir:
                        cmd.extend(['--config-dir', cfg_dir])
            except cfg.NoSuchOptError:
                pass
        cmd.extend(['--privsep_context', self.pypath, '--privsep_sock_path', sockpath])
        return cmd

    def set_client_mode(self, enabled):
        if enabled and sys.platform == 'win32':
            raise RuntimeError('Enabling the client_mode is not currently supported on Windows.')
        self.client_mode = enabled

    def entrypoint(self, func):
        """This is intended to be used as a decorator."""
        return self._entrypoint(func)

    def entrypoint_with_timeout(self, timeout):
        """This is intended to be used as a decorator with timeout."""

        def wrap(func):

            @functools.wraps(func)
            def inner(*args, **kwargs):
                f = self._entrypoint(func)
                return f(*args, _wrap_timeout=timeout, **kwargs)
            setattr(inner, _ENTRYPOINT_ATTR, self)
            return inner
        return wrap

    def _entrypoint(self, func):
        if not func.__module__.startswith(self.prefix):
            raise AssertionError('%r entrypoints must be below "%s"' % (self, self.prefix))
        if getattr(func, _ENTRYPOINT_ATTR, None) is not None:
            raise AssertionError('%r is already associated with another PrivContext' % func)
        f = functools.partial(self._wrap, func)
        setattr(f, _ENTRYPOINT_ATTR, self)
        return f

    def is_entrypoint(self, func):
        return getattr(func, _ENTRYPOINT_ATTR, None) is self

    def _wrap(self, func, *args, _wrap_timeout=None, **kwargs):
        if self.client_mode:
            name = '%s.%s' % (func.__module__, func.__name__)
            if self.channel is not None and (not self.channel.running):
                LOG.warning('RESTARTING PrivContext for %s', name)
                self.stop()
            if self.channel is None:
                self.start()
            r_call_timeout = _wrap_timeout or self.timeout
            return self.channel.remote_call(name, args, kwargs, r_call_timeout)
        else:
            return func(*args, **kwargs)

    def start(self, method=Method.ROOTWRAP):
        with self.start_lock:
            if self.channel is not None:
                LOG.warning('privsep daemon already running')
                return
            if method is Method.ROOTWRAP:
                channel = daemon.RootwrapClientChannel(context=self)
            elif method is Method.FORK:
                channel = daemon.ForkingClientChannel(context=self)
            else:
                raise ValueError('Unknown method: %s' % method)
            self.channel = channel

    def stop(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None