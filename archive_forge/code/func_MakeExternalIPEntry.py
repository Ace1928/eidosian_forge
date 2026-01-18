from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakeExternalIPEntry(messages, stateful_ip_dict):
    return messages.StatefulPolicyPreservedState.ExternalIPsValue.AdditionalProperty(key=stateful_ip_dict.get('interface-name', flags.STATEFUL_IP_DEFAULT_INTERFACE_NAME), value=_MakeNetworkIPForStatefulIP(messages, stateful_ip_dict))