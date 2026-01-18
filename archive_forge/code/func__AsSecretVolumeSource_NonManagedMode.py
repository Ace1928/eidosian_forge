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
def _AsSecretVolumeSource_NonManagedMode(self, resource):
    messages = resource.MessagesModule()
    out = messages.SecretVolumeSource(secretName=self._GetOrCreateAlias(resource))
    if self.secret_version != SpecialVersion.MOUNT_ALL:
        out.items.append(messages.KeyToPath(key=self.secret_version, path=self.secret_version))
    return out