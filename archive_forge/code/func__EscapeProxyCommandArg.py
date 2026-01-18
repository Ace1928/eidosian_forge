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
def _EscapeProxyCommandArg(s, env):
    """Returns s escaped such that it can be a ProxyCommand arg.

  Args:
    s: str, Argument to escape. Must be non-empty.
    env: Environment, data about the ssh client.

  Raises:
    BadCharacterError: If s contains a bad character.
  """
    for c in s:
        if not 32 <= ord(c) < 127:
            raise BadCharacterError("Special character %r (part of %r) couldn't be escaped for ProxyCommand" % (c, s))
    if env.suite is Suite.PUTTY:
        s = _EscapeWindowsArgvElement(s)
        s = _EscapePuttyBackslashPercent(s)
        return s
    return _EscapeForBash(s).replace('%', '%%')