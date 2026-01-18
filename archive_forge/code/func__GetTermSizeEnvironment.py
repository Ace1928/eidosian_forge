from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def _GetTermSizeEnvironment():
    """Returns the terminal x and y dimensions from the environment."""
    return (int(os.environ['COLUMNS']), int(os.environ['LINES']))