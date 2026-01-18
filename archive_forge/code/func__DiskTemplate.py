from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _DiskTemplate(self, args, kind):
    kwargs = {}
    if kind == 'root':
        kwargs['sizeGib'] = flags.GetRootVolumeSize(args)
    elif kind == 'main':
        kwargs['sizeGib'] = flags.GetMainVolumeSize(args)
    return self._messages.GoogleCloudGkemulticloudV1AzureDiskTemplate(**kwargs) if any(kwargs.values()) else None