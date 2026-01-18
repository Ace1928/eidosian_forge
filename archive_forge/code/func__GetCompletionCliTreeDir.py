from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _GetCompletionCliTreeDir():
    """Returns the SDK static completion CLI tree dir."""
    return os.path.join(_GetInstallationRootDir(), 'data', 'cli')