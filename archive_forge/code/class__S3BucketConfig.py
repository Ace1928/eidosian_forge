from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _S3BucketConfig(_BucketConfig):
    """Holder for S3-specific bucket fields.

  See superclass for attributes.
  We currently don't support any S3-only fields. This class exists to maintain
  the provider-specific subclass pattern used by the request config factory.
  """