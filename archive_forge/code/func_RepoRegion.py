from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def RepoRegion(args, cluster_location=None):
    """Returns the region for the Artifact Registry repo.

   The intended behavior is platform-specific:
   * managed: Same region as the service (run/region or --region)
   * gke: Appropriate region based on cluster zone (cluster_location arg)
   * kubernetes: The run/region config value will be used or an exception
     raised when unset.

  Args:
    args: Namespace, the args namespace.
    cluster_location: The zone which a Cloud Run for Anthos cluster resides.
      When specified, this will result in the region for this zone being
      returned.

  Returns:
    The appropriate region for the repository.
  """
    if cluster_location:
        return _RegionFromZone(cluster_location)
    region = flags.GetRegion(args, prompt=False)
    if region:
        return region
    raise exceptions.ArgumentError('To deploy from source with this platform, you must set run/region via "gcloud config set run/region REGION".')