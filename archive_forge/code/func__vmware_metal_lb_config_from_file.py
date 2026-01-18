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
def _vmware_metal_lb_config_from_file(self, args: parser_extensions.Namespace) -> messages.VmwareMetalLbConfig:
    file_content = args.metal_lb_config_from_file
    metal_lb_config = file_content.get('metalLBConfig', None)
    if not metal_lb_config:
        raise InvalidConfigFile('Missing field [metalLBConfig] in Metal LB configuration file.')
    address_pools = metal_lb_config.get('addressPools', [])
    if not address_pools:
        raise InvalidConfigFile('Missing field [addressPools] in Metal LB configuration file.')
    kwargs = {'addressPools': self._address_pools(address_pools)}
    return messages.VmwareMetalLbConfig(**kwargs)