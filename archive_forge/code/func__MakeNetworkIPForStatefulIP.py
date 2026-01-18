from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def _MakeNetworkIPForStatefulIP(messages, stateful_ip_dict):
    """Make NetworkIP proto out of stateful IP configuration dict."""
    network_ip = messages.StatefulPolicyPreservedStateNetworkIp()
    if stateful_ip_dict.get('auto-delete'):
        network_ip.autoDelete = stateful_ip_dict.get('auto-delete').GetAutoDeleteEnumValue(messages.StatefulPolicyPreservedStateNetworkIp.AutoDeleteValueValuesEnum)
    return network_ip