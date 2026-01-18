from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import errno
import getopt
import logging
import os
import re
import signal
import socket
import sys
import textwrap
import traceback
import six
from six.moves import configparser
from six.moves import range
from google.auth import exceptions as google_auth_exceptions
import gslib.exception
from gslib.exception import CommandException
from gslib.exception import ControlCException
from gslib.utils.version_check import check_python_version_support
from gslib.utils.arg_helper import GetArgumentsAndOptions
from gslib.utils.user_agent_helper import GetUserAgent
import boto
import gslib
from gslib.utils import system_util, text_util
from gslib import metrics
import httplib2
import oauth2client
from google_reauth import reauth_creds
from google_reauth import errors as reauth_errors
from gslib import context_config
from gslib import wildcard_iterator
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import ProjectIdException
from gslib.cloud_api import ServiceException
from gslib.command_runner import CommandRunner
import apitools.base.py.exceptions as apitools_exceptions
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import InitializeSignalHandling
from gslib.sig_handling import RegisterSignalHandler
def _HandleControlC(signal_num, cur_stack_frame):
    """Called when user hits ^C.

  This function prints a brief message instead of the normal Python stack trace
  (unless -D option is used).

  Args:
    signal_num: Signal that was caught.
    cur_stack_frame: Unused.
  """
    if debug_level >= 2:
        stack_trace = ''.join(traceback.format_list(traceback.extract_stack()))
        _OutputAndExit('DEBUG: Caught CTRL-C (signal %d) - Exception stack trace:\n    %s' % (signal_num, re.sub('\\n', '\n    ', stack_trace)), exception=ControlCException())
    else:
        _OutputAndExit('Caught CTRL-C (signal %d) - exiting' % signal_num, exception=ControlCException())