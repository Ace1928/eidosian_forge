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
def _CheckPrivateKeyFilePermissions(self, file_path):
    """Checks that the file has reasonable permissions for a private key.

    In particular, check that the filename provided by the user is not
    world- or group-readable. If either of these are true, we issue a warning
    and offer to fix the permissions.

    Args:
      file_path: The name of the private key file.
    """
    if system_util.IS_WINDOWS:
        return
    st = os.stat(file_path)
    if bool((stat.S_IRGRP | stat.S_IROTH) & st.st_mode):
        self.logger.warn('\nYour private key file is readable by people other than yourself.\nThis is a security risk, since anyone with this information can use your service account.\n')
        fix_it = input('Would you like gsutil to change the file permissions for you? (y/N) ')
        if fix_it in ('y', 'Y'):
            try:
                os.chmod(file_path, 256)
                self.logger.info('\nThe permissions on your file have been successfully modified.\nThe only access allowed is readability by the user (permissions 0400 in chmod).')
            except Exception as _:
                self.logger.warn('\nWe were unable to modify the permissions on your file.\nIf you would like to fix this yourself, consider running:\n"sudo chmod 400 </path/to/key>" for improved security.')
        else:
            self.logger.info('\nYou have chosen to allow this file to be readable by others.\nIf you would like to fix this yourself, consider running:\n"sudo chmod 400 </path/to/key>" for improved security.')