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
def _vmware_ip_block(self, ip_block):
    """Constructs proto message VmwareIpBlock."""
    gateway = ip_block.get('gateway', None)
    if not gateway:
        raise InvalidConfigFile('Missing field [gateway] in Static IP configuration file.')
    netmask = ip_block.get('netmask', None)
    if not netmask:
        raise InvalidConfigFile('Missing field [netmask] in Static IP configuration file.')
    host_ips = ip_block.get('ips', [])
    if not host_ips:
        raise InvalidConfigFile('Missing field [ips] in Static IP configuration file.')
    kwargs = {'gateway': gateway, 'netmask': netmask, 'ips': [self._vmware_host_ip(host_ip) for host_ip in host_ips]}
    if flags.IsSet(kwargs):
        return messages.VmwareIpBlock(**kwargs)
    return None