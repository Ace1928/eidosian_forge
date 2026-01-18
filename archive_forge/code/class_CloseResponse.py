import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
class CloseResponse:
    """Dummy empty response to trigger the no body status."""

    def __init__(self, close):
        """Use some defaults to ensure we have a header."""
        self.status = '200 OK'
        self.headers = {'Content-Type': 'text/html'}
        self.close = close

    def __getitem__(self, index):
        """Ensure we don't have a body."""
        raise IndexError()

    def output(self):
        """Return self to hook the close method."""
        return self