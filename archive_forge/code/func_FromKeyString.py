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
@classmethod
def FromKeyString(cls, key_string):
    """Construct a public key from a typical OpenSSH-style key string.

      Args:
        key_string: str, on the format `TYPE DATA [COMMENT]`. Example: `ssh-rsa
          ABCDEF me@host.com`.

      Raises:
        InvalidKeyError: The public key file does not contain key (heuristic).

      Returns:
        Keys.PublicKey, the parsed public key.
      """
    decoded_key = key_string.strip()
    if isinstance(key_string, six.binary_type):
        decoded_key = decoded_key.decode('utf-8', 'replace')
    parts = decoded_key.split(' ', 2)
    if len(parts) < 2:
        raise InvalidKeyError('Public key [{}] is invalid.'.format(key_string))
    comment = parts[2].strip() if len(parts) > 2 else ''
    return cls(parts[0], parts[1], comment)