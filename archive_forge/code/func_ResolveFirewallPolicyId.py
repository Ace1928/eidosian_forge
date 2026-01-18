from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.core import log
def ResolveFirewallPolicyId(firewall_policy, short_name, organization_id):
    """Returns the firewall policy id that matches the short_name in the org.

  Args:
    firewall_policy: the organization firewall policy.
    short_name: the short name of the firewall policy to be resolved.
    organization_id: the organization ID which the firewall policy belongs to.

  Returns:
    Firewall policy resource ID.
  """
    response = firewall_policy.List(parent_id=organization_id, only_generate_request=False)
    fp_id = None
    for fp in response[0].items:
        if fp.displayName == short_name:
            fp_id = fp.name
            break
    if fp_id is None:
        log.error('Invalid short name: {0}. No Security Policy with this short name exists.'.format(short_name))
        sys.exit()
    return fp_id