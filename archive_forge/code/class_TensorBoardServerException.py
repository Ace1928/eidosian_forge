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
class TensorBoardServerException(Exception):
    """Exception raised by TensorBoardServer for user-friendly errors.

    Subclasses of TensorBoardServer can raise this exception in order to
    generate a clean error message for the user rather than a
    stacktrace.
    """

    def __init__(self, msg):
        self.msg = msg