from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
class LintMiddleware:
    """Warns about common errors in the WSGI and HTTP behavior of the
    server and wrapped application. Some of the issues it checks are:

    -   invalid status codes
    -   non-bytes sent to the WSGI server
    -   strings returned from the WSGI application
    -   non-empty conditional responses
    -   unquoted etags
    -   relative URLs in the Location header
    -   unsafe calls to wsgi.input
    -   unclosed iterators

    Error information is emitted using the :mod:`warnings` module.

    :param app: The WSGI application to wrap.

    .. code-block:: python

        from werkzeug.middleware.lint import LintMiddleware
        app = LintMiddleware(app)
    """

    def __init__(self, app: WSGIApplication) -> None:
        self.app = app

    def check_environ(self, environ: WSGIEnvironment) -> None:
        if type(environ) is not dict:
            warn('WSGI environment is not a standard Python dict.', WSGIWarning, stacklevel=4)
        for key in ('REQUEST_METHOD', 'SERVER_NAME', 'SERVER_PORT', 'wsgi.version', 'wsgi.input', 'wsgi.errors', 'wsgi.multithread', 'wsgi.multiprocess', 'wsgi.run_once'):
            if key not in environ:
                warn(f'Required environment key {key!r} not found', WSGIWarning, stacklevel=3)
        if environ['wsgi.version'] != (1, 0):
            warn('Environ is not a WSGI 1.0 environ.', WSGIWarning, stacklevel=3)
        script_name = environ.get('SCRIPT_NAME', '')
        path_info = environ.get('PATH_INFO', '')
        if script_name and script_name[0] != '/':
            warn(f"'SCRIPT_NAME' does not start with a slash: {script_name!r}", WSGIWarning, stacklevel=3)
        if path_info and path_info[0] != '/':
            warn(f"'PATH_INFO' does not start with a slash: {path_info!r}", WSGIWarning, stacklevel=3)

    def check_start_response(self, status: str, headers: list[tuple[str, str]], exc_info: None | tuple[type[BaseException], BaseException, TracebackType]) -> tuple[int, Headers]:
        check_type('status', status, str)
        status_code_str = status.split(None, 1)[0]
        if len(status_code_str) != 3 or not status_code_str.isdecimal():
            warn('Status code must be three digits.', WSGIWarning, stacklevel=3)
        if len(status) < 4 or status[3] != ' ':
            warn(f'Invalid value for status {status!r}. Valid status strings are three digits, a space and a status explanation.', WSGIWarning, stacklevel=3)
        status_code = int(status_code_str)
        if status_code < 100:
            warn('Status code < 100 detected.', WSGIWarning, stacklevel=3)
        if type(headers) is not list:
            warn('Header list is not a list.', WSGIWarning, stacklevel=3)
        for item in headers:
            if type(item) is not tuple or len(item) != 2:
                warn('Header items must be 2-item tuples.', WSGIWarning, stacklevel=3)
            name, value = item
            if type(name) is not str or type(value) is not str:
                warn('Header keys and values must be strings.', WSGIWarning, stacklevel=3)
            if name.lower() == 'status':
                warn('The status header is not supported due to conflicts with the CGI spec.', WSGIWarning, stacklevel=3)
        if exc_info is not None and (not isinstance(exc_info, tuple)):
            warn('Invalid value for exc_info.', WSGIWarning, stacklevel=3)
        headers = Headers(headers)
        self.check_headers(headers)
        return (status_code, headers)

    def check_headers(self, headers: Headers) -> None:
        etag = headers.get('etag')
        if etag is not None:
            if etag.startswith(('W/', 'w/')):
                if etag.startswith('w/'):
                    warn('Weak etag indicator should be upper case.', HTTPWarning, stacklevel=4)
                etag = etag[2:]
            if not etag[:1] == etag[-1:] == '"':
                warn('Unquoted etag emitted.', HTTPWarning, stacklevel=4)
        location = headers.get('location')
        if location is not None:
            if not urlparse(location).netloc:
                warn('Absolute URLs required for location header.', HTTPWarning, stacklevel=4)

    def check_iterator(self, app_iter: t.Iterable[bytes]) -> None:
        if isinstance(app_iter, str):
            warn('The application returned a string. The response will send one character at a time to the client, which will kill performance. Return a list or iterable instead.', WSGIWarning, stacklevel=3)

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Iterable[bytes]:
        if len(args) != 2:
            warn('A WSGI app takes two arguments.', WSGIWarning, stacklevel=2)
        if kwargs:
            warn('A WSGI app does not take keyword arguments.', WSGIWarning, stacklevel=2)
        environ: WSGIEnvironment = args[0]
        start_response: StartResponse = args[1]
        self.check_environ(environ)
        environ['wsgi.input'] = InputStream(environ['wsgi.input'])
        environ['wsgi.errors'] = ErrorStream(environ['wsgi.errors'])
        environ['wsgi.file_wrapper'] = FileWrapper
        headers_set: list[t.Any] = []
        chunks: list[int] = []

        def checking_start_response(*args: t.Any, **kwargs: t.Any) -> t.Callable[[bytes], None]:
            if len(args) not in {2, 3}:
                warn(f'Invalid number of arguments: {len(args)}, expected 2 or 3.', WSGIWarning, stacklevel=2)
            if kwargs:
                warn("'start_response' does not take keyword arguments.", WSGIWarning)
            status: str = args[0]
            headers: list[tuple[str, str]] = args[1]
            exc_info: None | tuple[type[BaseException], BaseException, TracebackType] = args[2] if len(args) == 3 else None
            headers_set[:] = self.check_start_response(status, headers, exc_info)
            return GuardedWrite(start_response(status, headers, exc_info), chunks)
        app_iter = self.app(environ, t.cast('StartResponse', checking_start_response))
        self.check_iterator(app_iter)
        return GuardedIterator(app_iter, t.cast(t.Tuple[int, Headers], headers_set), chunks)