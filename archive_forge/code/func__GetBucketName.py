import os
import uuid
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def _GetBucketName(region):
    """Returns the default regional bucket name.

  Args:
    region: Cloud Run region.

  Returns:
    GCS bucket name.
  """
    safe_project = properties.VALUES.core.project.Get(required=True).replace(':', '_').replace('.', '_').replace('google', 'elgoog')
    return f'run-sources-{safe_project}-{region}' if region is not None else f'run-sources-{safe_project}'