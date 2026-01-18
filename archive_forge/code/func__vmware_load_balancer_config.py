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
def _vmware_load_balancer_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareLoadBalancerConfig."""
    kwargs = {'f5Config': self._vmware_f5_big_ip_config(args), 'metalLbConfig': self._vmware_metal_lb_config(args), 'manualLbConfig': self._vmware_manual_lb_config(args), 'vipConfig': self._vmware_vip_config(args)}
    if any(kwargs.values()):
        return messages.VmwareLoadBalancerConfig(**kwargs)
    return None