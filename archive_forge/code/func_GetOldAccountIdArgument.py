from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
def GetOldAccountIdArgument(positional=True):
    metavar = 'ACCOUNT_ID'
    help_ = 'Specify a billing account ID. Billing account IDs are of the form `0X0X0X-0X0X0X-0X0X0X`. To see available IDs, run `$ gcloud billing accounts list`.'
    if positional:
        return base.Argument('id', nargs='?', metavar=metavar, completer=BillingAccountsCompleter, action=actions.DeprecationAction('ACCOUNT_ID', show_message=lambda x: x is not None, removed=False, warn='The `{flag_name}` argument has been renamed `--billing-account`.'), help=help_)
    else:
        return base.Argument('--account-id', dest='billing_account', metavar=metavar, completer=BillingAccountsCompleter, action=actions.DeprecationAction('--account-id', removed=False, warn='The `{flag_name}` flag has been renamed `--billing-account`.'), help=help_)