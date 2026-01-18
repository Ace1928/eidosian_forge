from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vsphere_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareVsphereConfig."""
    if 'vsphere_config' not in args.GetSpecifiedArgsDict():
        return None
    kwargs = {'datastore': args.vsphere_config.get('datastore', None), 'storagePolicyName': args.vsphere_config.get('storage-policy-name', None)}
    if flags.IsSet(kwargs):
        return messages.VmwareVsphereConfig(**kwargs)
    return None