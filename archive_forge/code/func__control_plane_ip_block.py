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
def _control_plane_ip_block(self, args: parser_extensions.Namespace):
    """Constructs proto message ControlPlaneIpBlock."""
    if 'control_plane_ip_block' not in args.GetSpecifiedArgsDict():
        return None
    kwargs = {'gateway': args.control_plane_ip_block.get('gateway', None), 'netmask': args.control_plane_ip_block.get('netmask', None), 'ips': [messages.VmwareHostIp(ip=ip[0], hostname=ip[1]) for ip in args.control_plane_ip_block.get('ips', [])]}
    return messages.VmwareIpBlock(**kwargs)