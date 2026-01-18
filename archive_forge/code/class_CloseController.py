import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
class CloseController:
    """Controller for testing the close callback."""

    def __call__(self, environ, start_response):
        """Get the req to know header sent status."""
        self.req = start_response.__self__.req
        resp = CloseResponse(self.close)
        start_response(resp.status, resp.headers.items())
        return resp

    def close(self):
        """Close, writing hello."""
        self.req.write(b'hello')