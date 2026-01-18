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
def _run_serve_subcommand(self, flags):
    if flags.version_tb:
        print(version.VERSION)
        return 0
    if flags.inspect:
        logger.info('Not bringing up TensorBoard, but inspecting event files.')
        event_file = os.path.expanduser(flags.event_file)
        efi.inspect(flags.logdir, event_file, flags.tag)
        return 0
    try:
        server = self._make_server()
        server.print_serving_message()
        self._register_info(server)
        server.serve_forever()
        return 0
    except TensorBoardServerException as e:
        logger.error(e.msg)
        sys.stderr.write('ERROR: %s\n' % e.msg)
        sys.stderr.flush()
        return -1