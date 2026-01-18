from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _ControlPlane(self, args):
    control_plane_type = self._messages.GoogleCloudGkemulticloudV1AwsControlPlane
    kwargs = {'awsServicesAuthentication': self._ServicesAuthentication(args), 'configEncryption': self._ConfigEncryption(args), 'databaseEncryption': self._DatabaseEncryption(args), 'iamInstanceProfile': aws_flags.GetIamInstanceProfile(args), 'instancePlacement': self._InstancePlacement(args), 'instanceType': aws_flags.GetInstanceType(args), 'mainVolume': self._VolumeTemplate(args, 'main'), 'proxyConfig': self._ProxyConfig(args), 'rootVolume': self._VolumeTemplate(args, 'root'), 'securityGroupIds': aws_flags.GetSecurityGroupIds(args), 'sshConfig': self._SshConfig(args), 'subnetIds': aws_flags.GetSubnetIds(args), 'version': flags.GetClusterVersion(args), 'tags': self._Tags(args, control_plane_type)}
    return self._messages.GoogleCloudGkemulticloudV1AwsControlPlane(**kwargs) if any(kwargs.values()) else None