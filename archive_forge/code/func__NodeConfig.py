from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _NodeConfig(self, args):
    node_config_type = self._messages.GoogleCloudGkemulticloudV1AwsNodeConfig
    kwargs = {'configEncryption': self._ConfigEncryption(args), 'iamInstanceProfile': aws_flags.GetIamInstanceProfile(args), 'imageType': flags.GetImageType(args), 'instancePlacement': self._InstancePlacement(args), 'instanceType': aws_flags.GetInstanceType(args), 'proxyConfig': self._ProxyConfig(args), 'rootVolume': self._VolumeTemplate(args, 'root'), 'securityGroupIds': aws_flags.GetSecurityGroupIds(args), 'spotConfig': self._SpotConfig(args), 'sshConfig': self._SshConfig(args), 'taints': flags.GetNodeTaints(args), 'labels': self._Labels(args, node_config_type), 'tags': self._Tags(args, node_config_type), 'autoscalingMetricsCollection': self._AutoScalingMetricsCollection(args)}
    return self._messages.GoogleCloudGkemulticloudV1AwsNodeConfig(**kwargs) if any(kwargs.values()) else None