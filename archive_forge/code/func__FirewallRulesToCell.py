from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _FirewallRulesToCell(firewall):
    """Returns a compact string describing the firewall rules."""
    rules = []
    for allowed in firewall.get('allowed', []):
        protocol = allowed.get('IPProtocol')
        if not protocol:
            continue
        port_ranges = allowed.get('ports')
        if port_ranges:
            for port_range in port_ranges:
                rules.append('{0}:{1}'.format(protocol, port_range))
        else:
            rules.append(protocol)
    return ','.join(rules)