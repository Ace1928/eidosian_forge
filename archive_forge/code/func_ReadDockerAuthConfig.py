from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
def ReadDockerAuthConfig():
    """Retrieve the contents of the Docker authorization entry.

  NOTE: This is public only to facilitate testing.

  Returns:
    The map of authorizations used by docker.
  """
    path, new_format = client_lib.GetDockerConfigPath()
    structure = client_lib.ReadConfigurationFile(path)
    if new_format:
        return structure['auths'] if 'auths' in structure else {}
    else:
        return structure