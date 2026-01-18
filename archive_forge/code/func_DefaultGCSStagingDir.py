from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def DefaultGCSStagingDir(deployment_short_name, location):
    """Get default staging directory.

  Args:
    deployment_short_name: short name of the deployment.
    location: location of the deployment.

  Returns:
    A default staging directory string.
  """
    gcs_source_bucket_name = GetDefaultStagingBucket()
    gcs_source_staging_dir = 'gs://{0}/{1}/{2}/{3}'.format(gcs_source_bucket_name, STAGING_DIR, location, deployment_short_name)
    return gcs_source_staging_dir