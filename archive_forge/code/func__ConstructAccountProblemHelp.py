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
def _ConstructAccountProblemHelp(reason):
    """Constructs a help string for an access control error.

  Args:
    reason: e.reason string from caught exception.

  Returns:
    Contructed help text.
  """
    default_project_id = boto.config.get_value('GSUtil', 'default_project_id')
    acct_help = 'Your request resulted in an AccountProblem (403) error. Usually this happens if you attempt to create a bucket without first having enabled billing for the project you are using. Please ensure billing is enabled for your project by following the instructions at `Google Cloud Platform Console<https://support.google.com/cloud/answer/6158867>`. '
    if default_project_id:
        acct_help += 'In the project overview, ensure that the Project Number listed for your project matches the project ID (%s) from your boto config file. ' % default_project_id
    acct_help += "If the above doesn't resolve your AccountProblem, please send mail to buganizer-system+187143@google.com requesting assistance, noting the exact command you ran, the fact that you received a 403 AccountProblem error, and your project ID. Please do not post your project ID on StackOverflow. Note: It's possible to use Google Cloud Storage without enabling billing if you're only listing or reading objects for which you're authorized, or if you're uploading objects to a bucket billed to a project that has billing enabled. But if you're attempting to create buckets or upload objects to a bucket owned by your own project, you must first enable billing for that project."
    return acct_help