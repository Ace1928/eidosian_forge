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
def Poll(self, env=None, putty_force_connect=False):
    """Poll a remote for connectivity within the given timeout.

    The SSH command may prompt the user. It is recommended to wrap this call in
    a progress tracker. If this method returns, a connection was successfully
    established. If not, this method will raise.

    Args:
      env: Environment, environment to run in (or current if None).
      putty_force_connect: bool, whether to inject 'y' into the prompts for
        `plink`, which is insecure and not recommended. It serves legacy
        compatibility purposes for existing usages only; DO NOT SET THIS IN NEW
        CODE.

    Raises:
      MissingCommandError: If SSH command(s) not found.
      core.retry.WaitException: SSH command failed, possibly due to short
        timeout. There is no way to distinguish between a timeout error and a
        misconfigured connection.
    """
    self._retryer.RetryOnException(self.ssh_command.Run, kwargs={'env': env, 'putty_force_connect': putty_force_connect}, should_retry_if=lambda exc_type, *args: exc_type is CommandError, sleep_ms=self._sleep_ms)