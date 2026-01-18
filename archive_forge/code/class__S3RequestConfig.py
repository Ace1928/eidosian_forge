from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _S3RequestConfig(_RequestConfig):
    """Holder for S3-specific API request parameters.

  Currently just meant for use with S3ObjectConfig and S3BucketConfig in
  the parent class "resource_args" field.
  """