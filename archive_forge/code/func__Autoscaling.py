from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _Autoscaling(self, args):
    kwargs = {'minNodeCount': flags.GetMinNodes(args), 'maxNodeCount': flags.GetMaxNodes(args)}
    if not any(kwargs.values()):
        return None
    return self._messages.GoogleCloudGkemulticloudV1AzureNodePoolAutoscaling(**kwargs)