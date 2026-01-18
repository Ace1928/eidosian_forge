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
def GetFirewallPolicyOrganization(firewall_policy_client, firewall_policy_id, optional_organization):
    """Returns ID of the organization the given firewall policy belongs to.

  Args:
    firewall_policy_client: the organization firewall policy client.
    firewall_policy_id: the short name or ID of the firewall policy to be
      resolved.
    optional_organization: organization if provided.

  Returns:
    Firewall policy resource ID.
  """
    if not re.match('\\d{9,15}', firewall_policy_id) and optional_organization is None:
        raise exceptions.RequiredArgumentException('--organization', 'Must set --organization=ORGANIZATION when short name [{0}]is used.'.format(firewall_policy_id))
    organization = optional_organization
    if not organization:
        fetched_policies = firewall_policy_client.Describe(fp_id=firewall_policy_id)
        if not fetched_policies:
            raise compute_exceptions.InvalidResourceError('Firewall Policy [{0}] does not exist.'.format(firewall_policy_id))
        organization = fetched_policies[0].parent
    if '/' in organization:
        organization = organization.split('/')[1]
    return organization