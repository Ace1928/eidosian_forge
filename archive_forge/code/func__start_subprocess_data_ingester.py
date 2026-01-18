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
def _start_subprocess_data_ingester(self):
    """Creates, starts, and returns a `SubprocessServerDataIngester`."""
    flags = self.flags
    server_binary = server_ingester.get_server_binary()
    ingester = server_ingester.SubprocessServerDataIngester(server_binary=server_binary, logdir=flags.logdir, reload_interval=flags.reload_interval, channel_creds_type=flags.grpc_creds_type, samples_per_plugin=flags.samples_per_plugin, extra_flags=shlex.split(flags.extra_data_server_flags))
    ingester.start()
    return ingester