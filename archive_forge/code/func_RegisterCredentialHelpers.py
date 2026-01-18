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
def RegisterCredentialHelpers(self, mappings_dict=None):
    """Adds Docker 'credHelpers' entry to this configuration.

    Adds Docker 'credHelpers' entry to this configuration and writes updated
    configuration to disk.

    Args:
      mappings_dict: The dict of 'credHelpers' mappings ({registry: handler}) to
        add to the Docker configuration. If not set, use the values from
        BuildOrderedCredentialHelperRegistries(DefaultAuthenticatedRegistries())

    Raises:
      ValueError: mappings are not a valid dict.
      DockerConfigUpdateError: Configuration does not support 'credHelpers'.
    """
    mappings_dict = mappings_dict or BuildOrderedCredentialHelperRegistries(DefaultAuthenticatedRegistries())
    if not isinstance(mappings_dict, dict):
        raise ValueError('Invalid Docker credential helpers mappings {}'.format(mappings_dict))
    if not self.SupportsRegistryHelpers():
        raise DockerConfigUpdateError('Credential Helpers not supported for this Docker client version {}'.format(self.DockerVersion()))
    self.contents[CREDENTIAL_HELPER_KEY] = mappings_dict
    self.WriteToDisk()