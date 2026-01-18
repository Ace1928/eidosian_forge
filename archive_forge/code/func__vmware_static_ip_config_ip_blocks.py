from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator, Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.api_lib.container.vmware import version_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vmware_static_ip_config_ip_blocks(self, args: parser_extensions.Namespace):
    vmware_static_ip_config_message = messages.VmwareStaticIpConfig()
    for ip_block in args.static_ip_config_ip_blocks:
        vmware_ip_block_message = messages.VmwareIpBlock(gateway=ip_block['gateway'], netmask=ip_block['netmask'], ips=[messages.VmwareHostIp(ip=ip[0], hostname=ip[1]) for ip in ip_block['ips']])
        vmware_static_ip_config_message.ipBlocks.append(vmware_ip_block_message)
    return vmware_static_ip_config_message