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
def EnsureSSHKeyExists(self, compute_client, user, instance, project, expiration):
    """Controller for EnsureSSHKey* variants.

    Sends the key to the project metadata or instance metadata,
    and signals whether the key was newly added.

    Args:
      compute_client: The compute client.
      user: str, The user name.
      instance: Instance, the instance to connect to.
      project: Project, the project instance is in.
      expiration: datetime, If not None, the point after which the key is no
          longer valid.


    Returns:
      bool, True if the key was newly added.
    """
    _, ssh_legacy_keys = _GetSSHKeysFromMetadata(instance.metadata)
    if ssh_legacy_keys:
        keys_newly_added = self.EnsureSSHKeyIsInInstance(compute_client, user, instance, expiration, legacy=True)
    elif _MetadataHasBlockProjectSshKeys(instance.metadata):
        keys_newly_added = self.EnsureSSHKeyIsInInstance(compute_client, user, instance, expiration)
    else:
        try:
            keys_newly_added = self.EnsureSSHKeyIsInProject(compute_client, user, project, expiration)
        except SetProjectMetadataError:
            log.info('Could not set project metadata:', exc_info=True)
            log.info('Attempting to set instance metadata.')
            keys_newly_added = self.EnsureSSHKeyIsInInstance(compute_client, user, instance, expiration)
    return keys_newly_added