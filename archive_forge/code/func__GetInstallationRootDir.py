from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _GetInstallationRootDir():
    """Returns the SDK installation root dir."""
    return os.path.sep.join(__file__.split(os.path.sep)[:-5])