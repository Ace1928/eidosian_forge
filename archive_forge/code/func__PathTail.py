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
def _PathTail(self):
    """Last path component of self._connector."""
    if self._connector is SpecialConnector.PATH_OR_ENV:
        raise TypeError("Can't make SecretVolumeSource message for secret %r of unknown usage." % self.secret_name)
    if not self._connector.startswith('/'):
        raise TypeError("Can't make SecretVolumeSource message for secret connected to env var %r." % self._connector)
    return self._connector.rsplit('/', 1)[-1]