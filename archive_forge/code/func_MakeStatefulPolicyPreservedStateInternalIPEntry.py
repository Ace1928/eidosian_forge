from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakeStatefulPolicyPreservedStateInternalIPEntry(messages, stateful_ip_dict):
    """Make InternalIPsValue proto for a given stateful IP configuration dict."""
    return messages.StatefulPolicyPreservedState.InternalIPsValue.AdditionalProperty(key=stateful_ip_dict.get('interface-name', flags.STATEFUL_IP_DEFAULT_INTERFACE_NAME), value=_MakeNetworkIPForStatefulIP(messages, stateful_ip_dict))