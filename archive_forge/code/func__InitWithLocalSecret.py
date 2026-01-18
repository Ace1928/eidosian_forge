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
def _InitWithLocalSecret(self, flag_value, connector_name):
    """Init this ReachableSecret for a simple, non-remote secret.

    Args:
      flag_value: str. A secret identifier like 'sec1:latest'. See tests for
        other supported formats.
      connector_name: Union[str, PATH_OR_ENV]. An env var, a mount point, or
        PATH_OR_ENV. See __init__ docs.

    Raises:
      ValueError on flag value parse failure.
    """
    self.remote_project_number = None
    parts = flag_value.split(':')
    if len(parts) == 1:
        self.secret_name, = parts
        self.secret_version = self._OmittedSecretKeyDefault(connector_name)
    elif len(parts) == 2:
        self.secret_name, self.secret_version = parts
    else:
        raise ValueError('Invalid secret spec %r' % flag_value)
    self._AssertValidLocalSecretName(self.secret_name)