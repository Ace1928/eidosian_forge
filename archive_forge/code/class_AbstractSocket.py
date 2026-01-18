import argparse
import os
import sys
import urllib.parse  # noqa: WPS301
from importlib import import_module
from contextlib import suppress
from . import server
from . import wsgi
class AbstractSocket(BindLocation):
    """AbstractSocket."""

    def __init__(self, abstract_socket):
        """Initialize."""
        self.bind_addr = '\x00{sock_path}'.format(sock_path=abstract_socket)