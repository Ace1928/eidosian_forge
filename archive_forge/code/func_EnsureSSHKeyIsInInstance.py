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
def EnsureSSHKeyIsInInstance(self, client, user, instance, expiration, legacy=False):
    """Ensures that the user's public SSH key is in the instance metadata.

    Args:
      client: The compute client.
      user: str, the name of the user associated with the SSH key in the
          metadata
      instance: Instance, ensure the SSH key is in the metadata of this instance
      expiration: datetime, If not None, the point after which the key is no
          longer valid.
      legacy: If the key is not present in metadata, add it to the legacy
          metadata entry instead of the default entry.

    Returns:
      bool, True if the key was newly added, False if it was in the metadata
          already
    """
    public_key = self.keys.GetPublicKey()
    new_metadata = _AddSSHKeyToMetadataMessage(client.messages, user, public_key, instance.metadata, expiration=expiration, legacy=legacy)
    has_new_metadata = new_metadata != instance.metadata
    if has_new_metadata:
        self.SetInstanceMetadata(client, instance, new_metadata)
    return has_new_metadata