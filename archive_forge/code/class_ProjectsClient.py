from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.billing import utils
class ProjectsClient(object):
    """High-level client for billing projects service."""

    def __init__(self, client=None, messages=None):
        self.client = client or utils.GetClient()
        self.messages = messages or self.client.MESSAGES_MODULE

    def Get(self, project_ref):
        return self.client.projects.GetBillingInfo(self.messages.CloudbillingProjectsGetBillingInfoRequest(name=project_ref.RelativeName()))

    def Link(self, project_ref, account_ref):
        """Link the given account to the given project.

    Args:
      project_ref: a Resource reference to the project to be linked to
      account_ref: a Resource reference to the account to link, or None to
        unlink the project from its current account.

    Returns:
      ProjectBillingInfo, the new ProjectBillingInfo
    """
        billing_account_name = account_ref.RelativeName() if account_ref else ''
        return self.client.projects.UpdateBillingInfo(self.messages.CloudbillingProjectsUpdateBillingInfoRequest(name=project_ref.RelativeName(), projectBillingInfo=self.messages.ProjectBillingInfo(billingAccountName=billing_account_name)))

    def List(self, account_ref, limit=None):
        return list_pager.YieldFromList(self.client.billingAccounts_projects, self.messages.CloudbillingBillingAccountsProjectsListRequest(name=account_ref.RelativeName()), field='projectBillingInfo', batch_size_attribute='pageSize', limit=limit)