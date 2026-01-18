from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.oslogin import client
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
class DescribeProfile(base.Command):
    """Describe the OS Login profile for the current user."""

    def __init__(self, *args, **kwargs):
        super(DescribeProfile, self).__init__(*args, **kwargs)

    def Run(self, args):
        """See ssh_utils.BaseSSHCLICommand.Run."""
        oslogin_client = client.OsloginClient(self.ReleaseTrack())
        user_email = properties.VALUES.auth.impersonate_service_account.Get() or properties.VALUES.core.account.Get()
        return oslogin_client.GetLoginProfile(user_email)