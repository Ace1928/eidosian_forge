from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.oslogin import client
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class RemoveProfile(base.Command):
    """Remove the posix account information for the current user."""

    def Run(self, args):
        oslogin_client = client.OsloginClient(self.ReleaseTrack())
        account = properties.VALUES.auth.impersonate_service_account.Get() or properties.VALUES.core.account.GetOrFail()
        project = properties.VALUES.core.project.Get(required=True)
        project_ref = resources.REGISTRY.Parse(project, params={'user': account}, collection='oslogin.users.projects')
        current_profile = oslogin_client.GetLoginProfile(account)
        account_id = None
        for account in current_profile.posixAccounts:
            if account.accountId == project:
                account_id = account.accountId
        if account_id:
            console_io.PromptContinue('Posix accounts associated with project ID [{0}] will be deleted.'.format(project), default=True, cancel_on_no=True)
            operating_system = getattr(args, 'operating_system', None)
            res = oslogin_client.DeletePosixAccounts(project_ref, operating_system)
            log.DeletedResource(account_id, details='posix account(s)')
            return res
        else:
            log.warning('No profile found with accountId [{0}]'.format(project))