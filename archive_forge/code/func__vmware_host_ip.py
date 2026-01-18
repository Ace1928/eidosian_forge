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
def _vmware_host_ip(self, host_ip) -> messages.VmwareHostIp:
    """Constructs proto message VmwareHostIp."""
    hostname = host_ip.get('hostname', None)
    if not hostname:
        raise InvalidConfigFile('Missing field [hostname] in Static IP configuration file.')
    ip = host_ip.get('ip', None)
    if not ip:
        raise InvalidConfigFile('Missing field [ip] in Static IP configuration file.')
    kwargs = {'hostname': hostname, 'ip': ip}
    return messages.VmwareHostIp(**kwargs)