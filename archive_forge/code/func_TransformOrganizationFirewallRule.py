from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformOrganizationFirewallRule(rule, undefined=''):
    """Returns a compact string describing an organization firewall rule.

  The compact string is a comma-separated list of PROTOCOL:PORT_RANGE items.
  If a particular protocol has no port ranges then only the protocol is listed.

  Args:
    rule: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    A compact string describing the organizatin firewall rule in the rule.
  """
    protocol = resource_transform.GetKeyValue(rule, 'ipProtocol', None)
    if protocol is None:
        return undefined
    result = []
    port_ranges = resource_transform.GetKeyValue(rule, 'ports', None)
    try:
        for port_range in port_ranges:
            result.append('{0}:{1}'.format(protocol, port_range))
    except TypeError:
        result.append(protocol)
    return ','.join(result)