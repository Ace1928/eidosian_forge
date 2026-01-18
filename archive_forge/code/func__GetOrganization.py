from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import policies as policies_api
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.meta import cache_util as meta_cache_util
from googlecloudsdk.command_lib.util import cache_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@cache_util.CacheResource('organizations-by-domain', 10)
def _GetOrganization(domain):
    """Get the organization for the given domain.

  The current user must have permission to list the organization.

  Args:
    domain: str, the domain (e.g. 'example.com') to look up the organization of,
      or None to just list the organizations for the current account.

  Returns:
    resources.Resource, a reference to a cloudresourcemanager.organizations
      resource

  Raises:
    DefaultPolicyResolutionError: if the number of organizations matching the
      given domain is not exactly 1, or searching organizations fails.
  """
    filter_ = 'domain:' + domain
    try:
        orgs = list(organizations.Client().List(filter_=filter_, limit=2))
    except Exception as err:
        raise DefaultPolicyResolutionError('Unable to resolve organization for domain [{}]: {}'.format(domain, err))
    if not orgs:
        raise DefaultPolicyResolutionError('No matching organizations found for domain [{}].'.format(domain))
    elif len(orgs) > 1:
        raise DefaultPolicyResolutionError('Found more than one organization for domain [{}].\n{}'.format(domain, orgs))
    return resources.REGISTRY.Parse(orgs[0].name, collection='cloudresourcemanager.organizations')