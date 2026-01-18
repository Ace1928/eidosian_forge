from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import multiprocessing
import os
import signal
import socket
import stat
import sys
import textwrap
import time
import webbrowser
from six.moves import input
from six.moves.http_client import ResponseNotReady
import boto
from boto.provider import Provider
import gslib
from gslib.command import Command
from gslib.command import DEFAULT_TASK_ESTIMATION_THRESHOLD
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.cred_types import CredTypes
from gslib.exception import AbortException
from gslib.exception import CommandException
from gslib.metrics import CheckAndMaybePromptForAnalyticsEnabling
from gslib.sig_handling import RegisterSignalHandler
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils.hashing_helper import CHECK_HASH_ALWAYS
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_SKIP
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.parallelism_framework_util import ShouldProhibitMultiprocessing
from httplib2 import ServerNotFoundError
from oauth2client.client import HAS_CRYPTO
def _OpenConfigFile(self, file_path):
    """Creates and opens a configuration file for writing.

    The file is created with mode 0600, and attempts to open existing files will
    fail (the latter is important to prevent symlink attacks).

    It is the caller's responsibility to close the file.

    Args:
      file_path: Path of the file to be created.

    Returns:
      A writable file object for the opened file.

    Raises:
      CommandException: if an error occurred when opening the file (including
          when the file already exists).
    """
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOINHERIT'):
        flags |= os.O_NOINHERIT
    try:
        fd = os.open(file_path, flags, 384)
    except (OSError, IOError) as e:
        raise CommandException('Failed to open %s for writing: %s' % (file_path, e))
    return os.fdopen(fd, 'w')