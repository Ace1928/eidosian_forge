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
def _make_server(self):
    """Constructs the TensorBoard WSGI app and instantiates the server."""
    data_provider, deprecated_multiplexer = self._make_data_provider()
    app = application.TensorBoardWSGIApp(self.flags, self.plugin_loaders, data_provider, self.assets_zip_provider, deprecated_multiplexer)
    return self.server_class(app, self.flags)