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
def _register_info(self, server):
    """Write a TensorBoardInfo file and arrange for its cleanup.

        Args:
          server: The result of `self._make_server()`.
        """
    server_url = urllib.parse.urlparse(server.get_url())
    info = manager.TensorBoardInfo(version=version.VERSION, start_time=int(time.time()), port=server_url.port, pid=os.getpid(), path_prefix=self.flags.path_prefix, logdir=self.flags.logdir or self.flags.logdir_spec, db=self.flags.db, cache_key=self.cache_key)
    atexit.register(manager.remove_info_file)
    manager.write_info_file(info)