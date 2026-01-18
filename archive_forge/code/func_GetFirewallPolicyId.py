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
def GetFirewallPolicyId(firewall_policy_client, firewall_policy, organization=None):
    """Returns the firewall policy id that matches the short_name in the org.

  Args:
    firewall_policy_client: the organization firewall policy client.
    firewall_policy: the short name or ID of the firewall policy to be resolved.
    organization: the organization ID which the firewall policy belongs to.

  Returns:
    Firewall policy resource ID.
  """
    if not re.match('\\d{9,15}', firewall_policy):
        if organization is None:
            log.error('Must set --organization=ORGANIZATION when short name [%s]is used.', firewall_policy)
            sys.exit()
        return ResolveFirewallPolicyId(firewall_policy_client, firewall_policy, ORGANIZATION_PREFIX + organization)
    return firewall_policy