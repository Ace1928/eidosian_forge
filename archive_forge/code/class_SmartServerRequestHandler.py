import threading
from _thread import get_ident
from ... import branch as _mod_branch
from ... import debug, errors, osutils, registry, revision, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...lazy_import import lazy_import
from breezy.bzr import bzrdir
from breezy.bzr.bundle import serializer
import tempfile
class SmartServerRequestHandler:
    """Protocol logic for smart server.

    This doesn't handle serialization at all, it just processes requests and
    creates responses.
    """

    def __init__(self, backing_transport, commands, root_client_path, jail_root=None):
        """Constructor.

        :param backing_transport: a Transport to handle requests for.
        :param commands: a registry mapping command names to SmartServerRequest
            subclasses. e.g. breezy.transport.smart.vfs.vfs_commands.
        """
        self._backing_transport = backing_transport
        self._root_client_path = root_client_path
        self._commands = commands
        if jail_root is None:
            jail_root = backing_transport
        self._jail_root = jail_root
        self.response = None
        self.finished_reading = False
        self._command = None
        if 'hpss' in debug.debug_flags:
            self._request_start_time = osutils.perf_counter()
            self._thread_id = get_ident()

    def _trace(self, action, message, extra_bytes=None, include_time=False):
        if include_time:
            t = '%5.3fs ' % (osutils.perf_counter() - self._request_start_time)
        else:
            t = ''
        if extra_bytes is None:
            extra = ''
        else:
            extra = ' ' + repr(extra_bytes[:40])
            if len(extra) > 33:
                extra = extra[:29] + extra[-1] + '...'
        trace.mutter('%12s: [%s] %s%s%s' % (action, self._thread_id, t, message, extra))

    def accept_body(self, bytes):
        """Accept body data."""
        if self._command is None:
            return
        self._run_handler_code(self._command.do_chunk, (bytes,), {})
        if 'hpss' in debug.debug_flags:
            self._trace('accept body', '%d bytes' % (len(bytes),), bytes)

    def end_of_body(self):
        """No more body data will be received."""
        self._run_handler_code(self._command.do_end, (), {})
        self.finished_reading = True
        if 'hpss' in debug.debug_flags:
            self._trace('end of body', '', include_time=True)

    def _run_handler_code(self, callable, args, kwargs):
        """Run some handler specific code 'callable'.

        If a result is returned, it is considered to be the commands response,
        and finished_reading is set true, and its assigned to self.response.

        Any exceptions caught are translated and a response object created
        from them.
        """
        result = self._call_converting_errors(callable, args, kwargs)
        if result is not None:
            self.response = result
            self.finished_reading = True

    def _call_converting_errors(self, callable, args, kwargs):
        """Call callable converting errors to Response objects."""
        try:
            self._command.setup_jail()
            try:
                return callable(*args, **kwargs)
            finally:
                self._command.teardown_jail()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as err:
            err_struct = _translate_error(err)
            return FailedSmartServerResponse(err_struct)

    def headers_received(self, headers):
        if 'hpss' in debug.debug_flags:
            self._trace('headers', repr(headers))

    def args_received(self, args):
        cmd = args[0]
        args = args[1:]
        try:
            command = self._commands.get(cmd)
        except LookupError:
            if 'hpss' in debug.debug_flags:
                self._trace('hpss unknown request', cmd, repr(args)[1:-1])
            raise errors.UnknownSmartMethod(cmd)
        if 'hpss' in debug.debug_flags:
            from . import vfs
            if issubclass(command, vfs.VfsRequest):
                action = 'hpss vfs req'
            else:
                action = 'hpss request'
            self._trace(action, '{} {}'.format(cmd, repr(args)[1:-1]))
        self._command = command(self._backing_transport, self._root_client_path, self._jail_root)
        self._run_handler_code(self._command.execute, args, {})

    def end_received(self):
        if self._command is None:
            return
        self._run_handler_code(self._command.do_end, (), {})
        if 'hpss' in debug.debug_flags:
            self._trace('end', '', include_time=True)

    def post_body_error_received(self, error_args):
        pass