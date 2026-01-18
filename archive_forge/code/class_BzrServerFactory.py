import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
class BzrServerFactory:
    """Helper class for serve_bzr."""

    def __init__(self, userdir_expander=None, get_base_path=None):
        self.cleanups = []
        self.base_path = None
        self.backing_transport = None
        if userdir_expander is None:
            userdir_expander = os.path.expanduser
        self.userdir_expander = userdir_expander
        if get_base_path is None:
            get_base_path = _local_path_for_transport
        self.get_base_path = get_base_path

    def _expand_userdirs(self, path):
        """Translate /~/ or /~user/ to e.g. /home/foo, using
        self.userdir_expander (os.path.expanduser by default).

        If the translated path would fall outside base_path, or the path does
        not start with ~, then no translation is applied.

        If the path is inside, it is adjusted to be relative to the base path.

        e.g. if base_path is /home, and the expanded path is /home/joe, then
        the translated path is joe.
        """
        result = path
        if path.startswith('~'):
            expanded = self.userdir_expander(path)
            if not expanded.endswith('/'):
                expanded += '/'
            if expanded.startswith(self.base_path):
                result = expanded[len(self.base_path):]
        return result

    def _make_expand_userdirs_filter(self, transport):
        return pathfilter.PathFilteringServer(transport, self._expand_userdirs)

    def _make_backing_transport(self, transport):
        """Chroot transport, and decorate with userdir expander."""
        self.base_path = self.get_base_path(transport)
        chroot_server = chroot.ChrootServer(transport)
        chroot_server.start_server()
        self.cleanups.append(chroot_server.stop_server)
        transport = _mod_transport.get_transport_from_url(chroot_server.get_url())
        if self.base_path is not None:
            expand_userdirs = self._make_expand_userdirs_filter(transport)
            expand_userdirs.start_server()
            self.cleanups.append(expand_userdirs.stop_server)
            transport = _mod_transport.get_transport_from_url(expand_userdirs.get_url())
        self.transport = transport

    def _get_stdin_stdout(self):
        return (sys.stdin.buffer, sys.stdout.buffer)

    def _make_smart_server(self, host, port, inet, timeout):
        if timeout is None:
            c = config.GlobalStack()
            timeout = c.get('serve.client_timeout')
        if inet:
            stdin, stdout = self._get_stdin_stdout()
            smart_server = medium.SmartServerPipeStreamMedium(stdin, stdout, self.transport, timeout=timeout)
        else:
            if host is None:
                host = medium.BZR_DEFAULT_INTERFACE
            if port is None:
                port = medium.BZR_DEFAULT_PORT
            smart_server = SmartTCPServer(self.transport, client_timeout=timeout)
            smart_server.start_server(host, port)
            trace.note(gettext('listening on port: %s'), str(smart_server.port))
        self.smart_server = smart_server

    def _change_globals(self):
        from breezy import lockdir, ui
        old_factory = ui.ui_factory
        old_lockdir_timeout = lockdir._DEFAULT_TIMEOUT_SECONDS

        def restore_default_ui_factory_and_lockdir_timeout():
            ui.ui_factory = old_factory
            lockdir._DEFAULT_TIMEOUT_SECONDS = old_lockdir_timeout
        self.cleanups.append(restore_default_ui_factory_and_lockdir_timeout)
        ui.ui_factory = ui.SilentUIFactory()
        lockdir._DEFAULT_TIMEOUT_SECONDS = 0
        orig = signals.install_sighup_handler()

        def restore_signals():
            signals.restore_sighup_handler(orig)
        self.cleanups.append(restore_signals)

    def set_up(self, transport, host, port, inet, timeout):
        self._make_backing_transport(transport)
        self._make_smart_server(host, port, inet, timeout)
        self._change_globals()

    def tear_down(self):
        for cleanup in reversed(self.cleanups):
            cleanup()