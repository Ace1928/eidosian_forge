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
def _vmware_static_ip_config_from_file(self, args: parser_extensions.Namespace):
    file_content = args.static_ip_config_from_file
    static_ip_config = file_content.get('staticIPConfig', None)
    if not static_ip_config:
        raise InvalidConfigFile('Missing field [staticIPConfig] in Static IP configuration file.')
    ip_blocks = static_ip_config.get('ipBlocks', [])
    if not ip_blocks:
        raise InvalidConfigFile('Missing field [ipBlocks] in Static IP configuration file.')
    kwargs = {'ipBlocks': [self._vmware_ip_block(ip_block) for ip_block in ip_blocks]}
    if flags.IsSet(kwargs):
        return messages.VmwareStaticIpConfig(**kwargs)
    return None