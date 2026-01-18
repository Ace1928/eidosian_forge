from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def CreateFullServiceAccountNameFromId(account_id):
    if not account_id.isdigit():
        raise gcloud_exceptions.InvalidArgumentException('account_id', 'Account unique ID should be a number. Please double check your input and try again.')
    return 'projects/-/serviceAccounts/' + account_id