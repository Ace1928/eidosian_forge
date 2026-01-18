from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
import six
def ParseFirewallRule(client, priority):
    """Creates a resource path given a firewall rule priority.

  Args:
    client: AppengineFirewallApiClient, the API client for this release track.
    priority: str, the priority of the rule.

  Returns:
    The resource for the rule.

  """
    res = GetRegistry(client.ApiVersion()).Parse(six.text_type(ParsePriority(priority)), params={'appsId': client.project}, collection='appengine.apps.firewall.ingressRules')
    return res