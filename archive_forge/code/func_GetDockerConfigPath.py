from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
from six.moves import urllib
def GetDockerConfigPath(force_new=False):
    """Retrieve the path to Docker's configuration file, noting its format.

  Args:
    force_new: bool, whether to force usage of the new config file regardless
               of whether it exists (for testing).

  Returns:
    A tuple containing:
    -The path to Docker's configuration file, and
    -A boolean indicating whether it is in the new (1.7.0+) configuration format
  """
    new_path = os.path.join(_GetNewConfigDirectory(), 'config.json')
    if os.path.exists(new_path) or force_new:
        return (new_path, True)
    old_path = os.path.join(_GetUserHomeDir(), '.dockercfg')
    return (old_path, False)