from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def EnableIamAccountConfirmation(response, args):
    del response
    if args.command_path[len(args.command_path) - 3:] == [u'iam', u'service-accounts', u'enable']:
        log.status.Print('Enabled service account [{}].'.format(args.service_account))