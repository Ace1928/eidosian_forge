from abc import ABCMeta
from abc import abstractmethod
import argparse
import atexit
from collections import defaultdict
import errno
import logging
import mimetypes
import os
import shlex
import signal
import socket
import sys
import threading
import time
import urllib.parse
from absl import flags as absl_flags
from absl.flags import argparse_flags
from werkzeug import serving
from tensorboard import manager
from tensorboard import version
from tensorboard.backend import application
from tensorboard.backend.event_processing import data_ingester as local_ingester
from tensorboard.backend.event_processing import event_file_inspector as efi
from tensorboard.data import server_ingester
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
class TensorBoardServer(metaclass=ABCMeta):
    """Class for customizing TensorBoard WSGI app serving."""

    @abstractmethod
    def __init__(self, wsgi_app, flags):
        """Create a flag-configured HTTP server for TensorBoard's WSGI app.

        Args:
          wsgi_app: The TensorBoard WSGI application to create a server for.
          flags: argparse.Namespace instance of TensorBoard flags.
        """
        raise NotImplementedError()

    @abstractmethod
    def serve_forever(self):
        """Blocking call to start serving the TensorBoard server."""
        raise NotImplementedError()

    @abstractmethod
    def get_url(self):
        """Returns a URL at which this server should be reachable."""
        raise NotImplementedError()

    def print_serving_message(self):
        """Prints a user-friendly message prior to server start.

        This will be called just before `serve_forever`.
        """
        sys.stderr.write('TensorBoard %s at %s (Press CTRL+C to quit)\n' % (version.VERSION, self.get_url()))
        sys.stderr.flush()