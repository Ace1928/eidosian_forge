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
def _CredentialStoreConfigured():
    """Returns True if a credential store is specified in the docker config.

  Returns:
    True if a credential store is specified in the docker config.
    False if the config file does not exist or does not contain a
    'credsStore' key.
  """
    try:
        path, is_new_format = client_lib.GetDockerConfigPath()
        contents = client_lib.ReadConfigurationFile(path)
        if is_new_format:
            return _CREDENTIAL_STORE_KEY in contents
        else:
            return False
    except IOError:
        return False