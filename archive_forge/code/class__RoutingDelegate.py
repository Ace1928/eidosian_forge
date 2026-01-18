import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class _RoutingDelegate(httputil.HTTPMessageDelegate):

    def __init__(self, router: Router, server_conn: object, request_conn: httputil.HTTPConnection) -> None:
        self.server_conn = server_conn
        self.request_conn = request_conn
        self.delegate = None
        self.router = router

    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        assert isinstance(start_line, httputil.RequestStartLine)
        request = httputil.HTTPServerRequest(connection=self.request_conn, server_connection=self.server_conn, start_line=start_line, headers=headers)
        self.delegate = self.router.find_handler(request)
        if self.delegate is None:
            app_log.debug('Delegate for %s %s request not found', start_line.method, start_line.path)
            self.delegate = _DefaultMessageDelegate(self.request_conn)
        return self.delegate.headers_received(start_line, headers)

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        assert self.delegate is not None
        return self.delegate.data_received(chunk)

    def finish(self) -> None:
        assert self.delegate is not None
        self.delegate.finish()

    def on_connection_close(self) -> None:
        assert self.delegate is not None
        self.delegate.on_connection_close()