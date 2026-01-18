from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
def _WarnOrReadFirstKeyLine(self, path, kind):
    """Returns the first line from the key file path.

    A None return indicates an error and is always accompanied by a log.warning
    message.

    Args:
      path: The path of the file to read from.
      kind: The kind of key file, 'private' or 'public'.

    Returns:
      None (and prints a log.warning message) if the file does not exist, is not
      readable, or is empty. Otherwise returns the first line utf8 decoded.
    """
    try:
        with files.FileReader(path) as f:
            line = f.readline().strip()
            if line:
                return line
            msg = 'is empty'
            status = KeyFileStatus.BROKEN
    except files.MissingFileError:
        msg = 'does not exist'
        status = KeyFileStatus.ABSENT
    except files.Error:
        msg = 'is not readable'
        status = KeyFileStatus.BROKEN
    log.warning('The %s SSH key file for gcloud %s.', kind, msg)
    return status