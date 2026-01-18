from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
def GetAccountIdArgument(positional=True, required=False):
    metavar = 'ACCOUNT_ID'
    help_ = 'Specify a billing account ID. Billing account IDs are of the form `0X0X0X-0X0X0X-0X0X0X`. To see available IDs, run `$ gcloud billing accounts list`.'
    if positional:
        return base.Argument('account_id', metavar=metavar, completer=BillingAccountsCompleter, help=help_)
    else:
        return base.Argument('--billing-account', metavar=metavar, required=required, completer=BillingAccountsCompleter, help=help_)