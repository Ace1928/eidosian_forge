from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instance_groups.flags import AutoDeleteFlag
from googlecloudsdk.command_lib.compute.instance_groups.flags import STATEFUL_IP_DEFAULT_INTERFACE_NAME
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
def MakePreservedStateInternalNetworkIpEntry(messages, stateful_ip):
    return messages.PreservedState.InternalIPsValue.AdditionalProperty(key=stateful_ip.get('interface-name', STATEFUL_IP_DEFAULT_INTERFACE_NAME), value=_MakePreservedStateNetworkIpEntry(messages, stateful_ip))