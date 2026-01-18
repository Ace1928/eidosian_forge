from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _Authorization(self, args):
    admin_users = flags.GetAdminUsers(args)
    admin_groups = flags.GetAdminGroups(args)
    if not admin_users and (not admin_groups):
        return None
    kwargs = {}
    if admin_users:
        kwargs['adminUsers'] = [self._messages.GoogleCloudGkemulticloudV1AwsClusterUser(username=u) for u in admin_users]
    if admin_groups:
        kwargs['adminGroups'] = [self._messages.GoogleCloudGkemulticloudV1AwsClusterGroup(group=g) for g in admin_groups]
    if not any(kwargs.values()):
        return None
    return self._messages.GoogleCloudGkemulticloudV1AwsAuthorization(**kwargs)