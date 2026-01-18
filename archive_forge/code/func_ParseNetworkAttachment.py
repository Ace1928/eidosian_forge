from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseNetworkAttachment(network_attachment, fallback_region=None):
    """Parses a network attachment name using configuration properties for fallback.

  Args:
    network_attachment: str, the network attachment's ID, fully-qualified URL,
      or relative name
    fallback_region: str, the region to use if `networkAttachment` does not
      contain region information. If None, and `networkAttachment` does not
      contain region information, parsing will fail.

  Returns:
    googlecloudsdk.core.resources.Resource: a resource reference for the
    networkAttachment
  """
    params = {'project': GetProject}
    if fallback_region:
        params['region'] = lambda r=fallback_region: r
    return resources.REGISTRY.Parse(network_attachment, params=params, collection='compute.networkAttachments')