from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import zip
class MissingStagingBucketException(Exception):
    """Indicates that a staging bucket was not provided with a local path.

  It doesn't inherit from core.exceptions.Error because it should be caught and
  re-raised at the call site with an actionable message.
  """