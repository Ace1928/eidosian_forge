from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _NodePool(self, node_pool_ref, args):
    nodepool_type = self._messages.GoogleCloudGkemulticloudV1AwsNodePool
    kwargs = {'annotations': self._Annotations(args, nodepool_type), 'autoscaling': self._NodePoolAutoscaling(args), 'config': self._NodeConfig(args), 'maxPodsConstraint': self._MaxPodsConstraint(args), 'management': self._NodeManagement(args), 'name': node_pool_ref.awsNodePoolsId, 'subnetId': flags.GetSubnetID(args), 'updateSettings': self._UpdateSettings(args), 'version': flags.GetNodeVersion(args)}
    return self._messages.GoogleCloudGkemulticloudV1AwsNodePool(**kwargs) if any(kwargs.values()) else None