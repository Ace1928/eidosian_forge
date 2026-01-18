from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _ClusterNetworking(self, args):
    kwargs = {'podAddressCidrBlocks': flags.GetPodAddressCidrBlocks(args), 'serviceAddressCidrBlocks': flags.GetServiceAddressCidrBlocks(args), 'vpcId': aws_flags.GetVpcId(args), 'perNodePoolSgRulesDisabled': aws_flags.GetPerNodePoolSGRulesDisabled(args)}
    return self._messages.GoogleCloudGkemulticloudV1AwsClusterNetworking(**kwargs) if any((x is not None for x in kwargs.values())) else None