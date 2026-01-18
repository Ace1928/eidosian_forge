import base64
from contextlib import closing
import gzip
from http.server import BaseHTTPRequestHandler
import os
import socket
from socketserver import ThreadingMixIn
import ssl
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import (
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from .openmetrics import exposition as openmetrics
from .registry import CollectorRegistry, REGISTRY
from .utils import floatToGoString
from .asgi import make_asgi_app  # noqa
class _PrometheusRedirectHandler(HTTPRedirectHandler):
    """
    Allow additional methods (e.g. PUT) and data forwarding in redirects.

    Use of this class constitute a user's explicit agreement to the
    redirect responses the Prometheus client will receive when using it.
    You should only use this class if you control or otherwise trust the
    redirect behavior involved and are certain it is safe to full transfer
    the original request (method and data) to the redirected URL. For
    example, if you know there is a cosmetic URL redirect in front of a
    local deployment of a Prometheus server, and all redirects are safe,
    this is the class to use to handle redirects in that case.

    The standard HTTPRedirectHandler does not forward request data nor
    does it allow redirected PUT requests (which Prometheus uses for some
    operations, for example `push_to_gateway`) because these cannot
    generically guarantee no violations of HTTP RFC 2616 requirements for
    the user to explicitly confirm redirects that could have unexpected
    side effects (such as rendering a PUT request non-idempotent or
    creating multiple resources not named in the original request).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """
        Apply redirect logic to a request.

        See parent HTTPRedirectHandler.redirect_request for parameter info.

        If the redirect is disallowed, this raises the corresponding HTTP error.
        If the redirect can't be determined, return None to allow other handlers
        to try. If the redirect is allowed, return the new request.

        This method specialized for the case when (a) the user knows that the
        redirect will not cause unacceptable side effects for any request method,
        and (b) the user knows that any request data should be passed through to
        the redirect. If either condition is not met, this should not be used.
        """
        m = getattr(req, 'method', req.get_method())
        if not (code in (301, 302, 303, 307) and m in ('GET', 'HEAD') or (code in (301, 302, 303) and m in ('POST', 'PUT'))):
            raise HTTPError(req.full_url, code, msg, headers, fp)
        new_request = Request(newurl.replace(' ', '%20'), headers=req.headers, origin_req_host=req.origin_req_host, unverifiable=True, data=req.data)
        new_request.method = m
        return new_request