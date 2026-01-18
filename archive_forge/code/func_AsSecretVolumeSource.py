from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def AsSecretVolumeSource(self, resource):
    """Build message for adding to resource.template.volumes.secrets.

    Args:
      resource: k8s_object that may get modified with new aliases.

    Returns:
      messages.SecretVolumeSource
    """
    if platforms.IsManaged():
        return self._AsSecretVolumeSource_ManagedMode(resource)
    else:
        return self._AsSecretVolumeSource_NonManagedMode(resource)