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
def _GetSSHKeysFromMetadata(metadata):
    """Returns the ssh-keys and legacy sshKeys metadata values.

  This function will return all of the SSH keys in metadata, stored in
  the default metadata entry ('ssh-keys') and the legacy entry ('sshKeys').

  Args:
    metadata: An instance or project metadata object.

  Returns:
    A pair of lists containing the SSH public keys in the default and
    legacy metadata entries.
  """
    ssh_keys = []
    ssh_legacy_keys = []
    if not metadata:
        return (ssh_keys, ssh_legacy_keys)
    for item in metadata.items:
        if item.key == constants.SSH_KEYS_METADATA_KEY:
            ssh_keys = _GetSSHKeyListFromMetadataEntry(item.value)
        elif item.key == constants.SSH_KEYS_LEGACY_METADATA_KEY:
            ssh_legacy_keys = _GetSSHKeyListFromMetadataEntry(item.value)
    return (ssh_keys, ssh_legacy_keys)