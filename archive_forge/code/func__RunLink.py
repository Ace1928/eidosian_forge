from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.billing import billing_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.billing import flags
from googlecloudsdk.command_lib.billing import utils
def _RunLink(args):
    client = billing_client.ProjectsClient()
    project_ref = utils.ParseProject(args.project_id)
    account_ref = utils.ParseAccount(args.billing_account)
    return client.Link(project_ref, account_ref)