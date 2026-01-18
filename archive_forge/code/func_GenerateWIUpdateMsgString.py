from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def GenerateWIUpdateMsgString(membership, issuer_url, resource_name, cluster_name):
    """Generates user message with information about enabling/disabling Workload Identity.

  We do not allow updating issuer url from one non-empty value to another.
  Args:
    membership: membership resource.
    issuer_url: The discovery URL for the cluster's service account token
      issuer.
    resource_name: The full membership resource name.
    cluster_name: User supplied cluster_name.

  Returns:
    A string, the message string for user to display information about
    enabling/disabling WI on a membership, if the issuer url is changed
    from empty to non-empty value or vice versa. An empty string is returned
    for other cases
  """
    if membership.authority and (not issuer_url):
        return 'A membership [{}] for the cluster [{}] already exists. The cluster was previously registered with Workload Identity enabled. Continuing will disable Workload Identity on your membership, and will reinstall the Connect agent deployment.'.format(resource_name, cluster_name)
    if not membership.authority and issuer_url:
        return 'A membership [{}] for the cluster [{}] already exists. The cluster was previously registered without Workload Identity. Continuing will enable Workload Identity on your membership, and will reinstall the Connect agent deployment.'.format(resource_name, cluster_name)
    return ''