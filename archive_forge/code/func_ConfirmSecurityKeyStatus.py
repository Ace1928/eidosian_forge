from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def ConfirmSecurityKeyStatus(oslogin_state):
    """Check the OS Login security key state and take approprate action.

  If OS Login security keys are not enabled, continue.
  When security keys are enabled:
    - if no security keys are configured in the user's account, show an error.
    - if the local SSH client doesn't support them, show an error.
    - if the user is using Putty, show an error.
    - if we cannnot determine if the local client supports security keys, show
      a warning and continue.

  Args:
    oslogin_state: An OsloginState object.

  Raises:
    SecurityKeysNotPresentError: If no security keys are registered in the
      user's account.
    SecurityKeysNotSupportedError: If the user's SSH client does not support
      security keys.

  Returns:
    None if no errors are raised.
  """
    if not oslogin_state.security_keys_enabled:
        return
    if not oslogin_state.security_keys:
        raise SecurityKeysNotPresentError('Instance requires security key for connection, but no security keys are registered in Google account.')
    if oslogin_state.ssh_security_key_support:
        return
    if oslogin_state.ssh_security_key_support is None:
        log.warning('Instance requires security key for connection, but cannot determine if the SSH client supports security keys. The connection may fail.')
        return
    if oslogin_state.environment == 'putty':
        raise SecurityKeysNotSupportedError('Instance requires security key for connection, but security keys are not supported on Windows using the PuTTY client.')
    raise SecurityKeysNotSupportedError('Instance requires security key for connection, but security keys are not supported by the installed SSH version. OpenSSH 8.4 or higher is required.')