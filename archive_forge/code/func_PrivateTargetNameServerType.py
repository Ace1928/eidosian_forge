from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def PrivateTargetNameServerType(value, version='v1'):
    """Build a single PrivateTargetNameServer based on 'value'.

  Args:
    value: (str) A string representation of an IPV4 ip address representing the
      PrivateTargetNameServer.
    version: (str) A string indicating the version of the API to be used, should
      be 'v1' only before removing BetaPrivateTargetNameServerType.

  Returns:
    A messages.PolicyAlternativeNameServerConfigTargetNameServer instance
    populated from the given ipv4 ip address.
  """
    messages = GetMessages(version)
    return messages.PolicyAlternativeNameServerConfigTargetNameServer(ipv4Address=value, forwardingPath=messages.PolicyAlternativeNameServerConfigTargetNameServer.ForwardingPathValueValuesEnum(1))