import logging
import re
import webob.dec
import webob.exc
from oslo_middleware import base
class CatchErrors(base.ConfigurableMiddleware):
    """Middleware that provides high-level error handling.

    It catches all exceptions from subsequent applications in WSGI pipeline
    to hide internal errors from API response.
    """

    @webob.dec.wsgify
    def __call__(self, req):
        try:
            response = req.get_response(self.application)
        except Exception:
            req_str = _TOKEN_RE.sub('\\1: *****', req.as_text())
            LOG.exception('An error occurred during processing the request: %s', req_str)
            response = webob.exc.HTTPInternalServerError()
        return response