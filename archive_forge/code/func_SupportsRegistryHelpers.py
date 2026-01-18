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
def SupportsRegistryHelpers(self):
    """Returns True unless Docker is confirmed to not support helpers."""
    try:
        return self.DockerVersion() >= MIN_DOCKER_CONFIG_HELPER_VERSION
    except:
        return True