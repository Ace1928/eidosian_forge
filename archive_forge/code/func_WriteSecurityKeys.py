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
def WriteSecurityKeys(oslogin_state):
    """Writes temporary files from a list of key data.

  Args:
    oslogin_state: An OsloginState object.

  Returns:
    List of file paths for security keys or None if security keys are not
    supported.
  """
    if oslogin_state.environment == 'putty':
        return None
    security_keys = oslogin_state.security_keys
    if not security_keys:
        return None
    key_dir = os.path.realpath(files.ExpandHomeDir(SECURITY_KEY_DIR))
    files.MakeDir(key_dir, mode=448)
    key_files = []
    for filename in os.listdir(key_dir):
        if filename.startswith('tmp_sk'):
            file_path = os.path.join(key_dir, filename)
            os.remove(file_path)
    for num, key in enumerate(security_keys):
        filename = 'tmp_sk_{0}'.format(num)
        file_path = os.path.join(key_dir, filename)
        files.WriteFileContents(file_path, key, private=True)
        key_files.append(file_path)
    return key_files