from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseSubnetwork(subnetwork, fallback_region=None):
    """Parses a subnetwork name using configuration properties for fallback.

  Args:
    subnetwork: str, the subnetwork's ID, fully-qualified URL, or relative name
    fallback_region: str, the region to use if `subnetwork` does not contain
        region information. If None, and `subnetwork` does not contain region
        information, parsing will fail.

  Returns:
    googlecloudsdk.core.resources.Resource: a resource reference for the
    subnetwork
  """
    params = {'project': GetProject}
    if fallback_region:
        params['region'] = lambda r=fallback_region: r
    return resources.REGISTRY.Parse(subnetwork, params=params, collection='compute.subnetworks')