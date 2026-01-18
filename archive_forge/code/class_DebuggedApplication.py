from __future__ import annotations
import getpass
import hashlib
import json
import os
import pkgutil
import re
import sys
import time
import typing as t
import uuid
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from os.path import basename
from os.path import join
from zlib import adler32
from .._internal import _log
from ..exceptions import NotFound
from ..http import parse_cookie
from ..security import gen_salt
from ..utils import send_file
from ..wrappers.request import Request
from ..wrappers.response import Response
from .console import Console
from .tbtools import DebugFrameSummary
from .tbtools import DebugTraceback
from .tbtools import render_console_html
class DebuggedApplication:
    """Enables debugging support for a given application::

        from werkzeug.debug import DebuggedApplication
        from myapp import app
        app = DebuggedApplication(app, evalex=True)

    The ``evalex`` argument allows evaluating expressions in any frame
    of a traceback. This works by preserving each frame with its local
    state. Some state, such as context globals, cannot be restored with
    the frame by default. When ``evalex`` is enabled,
    ``environ["werkzeug.debug.preserve_context"]`` will be a callable
    that takes a context manager, and can be called multiple times.
    Each context manager will be entered before evaluating code in the
    frame, then exited again, so they can perform setup and cleanup for
    each call.

    :param app: the WSGI application to run debugged.
    :param evalex: enable exception evaluation feature (interactive
                   debugging).  This requires a non-forking server.
    :param request_key: The key that points to the request object in this
                        environment.  This parameter is ignored in current
                        versions.
    :param console_path: the URL for a general purpose console.
    :param console_init_func: the function that is executed before starting
                              the general purpose console.  The return value
                              is used as initial namespace.
    :param show_hidden_frames: by default hidden traceback frames are skipped.
                               You can show them by setting this parameter
                               to `True`.
    :param pin_security: can be used to disable the pin based security system.
    :param pin_logging: enables the logging of the pin system.

    .. versionchanged:: 2.2
        Added the ``werkzeug.debug.preserve_context`` environ key.
    """
    _pin: str
    _pin_cookie: str

    def __init__(self, app: WSGIApplication, evalex: bool=False, request_key: str='werkzeug.request', console_path: str='/console', console_init_func: t.Callable[[], dict[str, t.Any]] | None=None, show_hidden_frames: bool=False, pin_security: bool=True, pin_logging: bool=True) -> None:
        if not console_init_func:
            console_init_func = None
        self.app = app
        self.evalex = evalex
        self.frames: dict[int, DebugFrameSummary | _ConsoleFrame] = {}
        self.frame_contexts: dict[int, list[t.ContextManager[None]]] = {}
        self.request_key = request_key
        self.console_path = console_path
        self.console_init_func = console_init_func
        self.show_hidden_frames = show_hidden_frames
        self.secret = gen_salt(20)
        self._failed_pin_auth = 0
        self.pin_logging = pin_logging
        if pin_security:
            if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' and pin_logging:
                _log('warning', ' * Debugger is active!')
                if self.pin is None:
                    _log('warning', ' * Debugger PIN disabled. DEBUGGER UNSECURED!')
                else:
                    _log('info', ' * Debugger PIN: %s', self.pin)
        else:
            self.pin = None

    @property
    def pin(self) -> str | None:
        if not hasattr(self, '_pin'):
            pin_cookie = get_pin_and_cookie_name(self.app)
            self._pin, self._pin_cookie = pin_cookie
        return self._pin

    @pin.setter
    def pin(self, value: str) -> None:
        self._pin = value

    @property
    def pin_cookie_name(self) -> str:
        """The name of the pin cookie."""
        if not hasattr(self, '_pin_cookie'):
            pin_cookie = get_pin_and_cookie_name(self.app)
            self._pin, self._pin_cookie = pin_cookie
        return self._pin_cookie

    def debug_application(self, environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterator[bytes]:
        """Run the application and conserve the traceback frames."""
        contexts: list[t.ContextManager[t.Any]] = []
        if self.evalex:
            environ['werkzeug.debug.preserve_context'] = contexts.append
        app_iter = None
        try:
            app_iter = self.app(environ, start_response)
            yield from app_iter
            if hasattr(app_iter, 'close'):
                app_iter.close()
        except Exception as e:
            if hasattr(app_iter, 'close'):
                app_iter.close()
            tb = DebugTraceback(e, skip=1, hide=not self.show_hidden_frames)
            for frame in tb.all_frames:
                self.frames[id(frame)] = frame
                self.frame_contexts[id(frame)] = contexts
            is_trusted = bool(self.check_pin_trust(environ))
            html = tb.render_debugger_html(evalex=self.evalex, secret=self.secret, evalex_trusted=is_trusted)
            response = Response(html, status=500, mimetype='text/html')
            try:
                yield from response(environ, start_response)
            except Exception:
                environ['wsgi.errors'].write('Debugging middleware caught exception in streamed response at a point where response headers were already sent.\n')
            environ['wsgi.errors'].write(''.join(tb.render_traceback_text()))

    def execute_command(self, request: Request, command: str, frame: DebugFrameSummary | _ConsoleFrame) -> Response:
        """Execute a command in a console."""
        contexts = self.frame_contexts.get(id(frame), [])
        with ExitStack() as exit_stack:
            for cm in contexts:
                exit_stack.enter_context(cm)
            return Response(frame.eval(command), mimetype='text/html')

    def display_console(self, request: Request) -> Response:
        """Display a standalone shell."""
        if 0 not in self.frames:
            if self.console_init_func is None:
                ns = {}
            else:
                ns = dict(self.console_init_func())
            ns.setdefault('app', self.app)
            self.frames[0] = _ConsoleFrame(ns)
        is_trusted = bool(self.check_pin_trust(request.environ))
        return Response(render_console_html(secret=self.secret, evalex_trusted=is_trusted), mimetype='text/html')

    def get_resource(self, request: Request, filename: str) -> Response:
        """Return a static resource from the shared folder."""
        path = join('shared', basename(filename))
        try:
            data = pkgutil.get_data(__package__, path)
        except OSError:
            return NotFound()
        else:
            if data is None:
                return NotFound()
            etag = str(adler32(data) & 4294967295)
            return send_file(BytesIO(data), request.environ, download_name=filename, etag=etag)

    def check_pin_trust(self, environ: WSGIEnvironment) -> bool | None:
        """Checks if the request passed the pin test.  This returns `True` if the
        request is trusted on a pin/cookie basis and returns `False` if not.
        Additionally if the cookie's stored pin hash is wrong it will return
        `None` so that appropriate action can be taken.
        """
        if self.pin is None:
            return True
        val = parse_cookie(environ).get(self.pin_cookie_name)
        if not val or '|' not in val:
            return False
        ts_str, pin_hash = val.split('|', 1)
        try:
            ts = int(ts_str)
        except ValueError:
            return False
        if pin_hash != hash_pin(self.pin):
            return None
        return time.time() - PIN_TIME < ts

    def _fail_pin_auth(self) -> None:
        time.sleep(5.0 if self._failed_pin_auth > 5 else 0.5)
        self._failed_pin_auth += 1

    def pin_auth(self, request: Request) -> Response:
        """Authenticates with the pin."""
        exhausted = False
        auth = False
        trust = self.check_pin_trust(request.environ)
        pin = t.cast(str, self.pin)
        bad_cookie = False
        if trust is None:
            self._fail_pin_auth()
            bad_cookie = True
        elif trust:
            auth = True
        elif self._failed_pin_auth > 10:
            exhausted = True
        else:
            entered_pin = request.args['pin']
            if entered_pin.strip().replace('-', '') == pin.replace('-', ''):
                self._failed_pin_auth = 0
                auth = True
            else:
                self._fail_pin_auth()
        rv = Response(json.dumps({'auth': auth, 'exhausted': exhausted}), mimetype='application/json')
        if auth:
            rv.set_cookie(self.pin_cookie_name, f'{int(time.time())}|{hash_pin(pin)}', httponly=True, samesite='Strict', secure=request.is_secure)
        elif bad_cookie:
            rv.delete_cookie(self.pin_cookie_name)
        return rv

    def log_pin_request(self) -> Response:
        """Log the pin if needed."""
        if self.pin_logging and self.pin is not None:
            _log('info', ' * To enable the debugger you need to enter the security pin:')
            _log('info', ' * Debugger pin code: %s', self.pin)
        return Response('')

    def __call__(self, environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterable[bytes]:
        """Dispatch the requests."""
        request = Request(environ)
        response = self.debug_application
        if request.args.get('__debugger__') == 'yes':
            cmd = request.args.get('cmd')
            arg = request.args.get('f')
            secret = request.args.get('s')
            frame = self.frames.get(request.args.get('frm', type=int))
            if cmd == 'resource' and arg:
                response = self.get_resource(request, arg)
            elif cmd == 'pinauth' and secret == self.secret:
                response = self.pin_auth(request)
            elif cmd == 'printpin' and secret == self.secret:
                response = self.log_pin_request()
            elif self.evalex and cmd is not None and (frame is not None) and (self.secret == secret) and self.check_pin_trust(environ):
                response = self.execute_command(request, cmd, frame)
        elif self.evalex and self.console_path is not None and (request.path == self.console_path):
            response = self.display_console(request)
        return response(environ, start_response)