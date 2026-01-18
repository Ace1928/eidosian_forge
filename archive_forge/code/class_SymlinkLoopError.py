from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.util import glob
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import map  # pylint: disable=redefined-builtin
class SymlinkLoopError(exceptions.Error):
    """Error indicating that there is a symlink loop."""