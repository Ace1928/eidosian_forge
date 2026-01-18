from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core.docker import client_lib as client_utils
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import semver
import six
def GetRegisteredCredentialHelpers(self):
    """Returns credential helpers entry from the Docker config file.

    Returns:
      'credHelpers' entry if it is specified in the Docker configuration or
      empty dict if the config does not contain a 'credHelpers' key.

    """
    if self.contents and CREDENTIAL_HELPER_KEY in self.contents:
        return {CREDENTIAL_HELPER_KEY: self.contents[CREDENTIAL_HELPER_KEY]}
    return {}