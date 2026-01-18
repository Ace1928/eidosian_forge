from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.core import log
def ResolveOrganizationSecurityPolicyId(org_security_policy, display_name, organization_id):
    """Returns the security policy id that matches the display_name in the org.

  Args:
    org_security_policy: the organization security policy.
    display_name: the display name of the security policy to be resolved.
    organization_id: the organization ID which the security policy belongs to.

  Returns:
    Security policy resource ID.
  """
    response = org_security_policy.List(parent_id=organization_id, only_generate_request=False)
    sp_id = None
    for sp in response[0].items:
        if sp.displayName == display_name:
            sp_id = sp.name
            break
    if sp_id is None:
        log.error('Invalid display name: {0}. No Security Policy with this display name exists.'.format(display_name))
        sys.exit()
    return sp_id