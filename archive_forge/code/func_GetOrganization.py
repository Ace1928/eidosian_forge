from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def GetOrganization(org_argument):
    """Get the Organization object for the provided Organization argument.

  Returns the organization object for a given organization ID or will search
  for and return the organization object associated with the given domain name.

  Args:
    org_argument: The value of the organization argument.

  Returns:
    An object representing an organization, or None if the organization could
    not be determined.
  """
    orgs_client = organizations.Client()
    org_id = StripOrgPrefix(org_argument)
    if org_id.isdigit():
        return orgs_client.Get(org_id)
    else:
        return orgs_client.GetByDomain(org_id)