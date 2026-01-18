from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def DeriveRegionFromLocation(location):
    """Returns the region from a location string.

  Args:
    location: A str of the form `<region>-<zone>` or `<region>`, such as
      `us-central1-a` or `us-central1`. Any other form will cause undefined
      behavior.

  Returns:
    The str region. Example: `us-central1`.
  """
    splits = location.split('-')
    return '-'.join(splits[:2])