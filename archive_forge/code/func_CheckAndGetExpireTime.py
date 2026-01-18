from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner.resource_args import CloudKmsKeyName
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.credentials import http
from googlecloudsdk.core.util import times
import six
from six.moves import http_client as httplib
from six.moves import urllib
def CheckAndGetExpireTime(args):
    """Check if fields for expireTime are correctly specified and parse value."""
    if args.IsSpecified('expiration_date') and args.IsSpecified('retention_period') or not (args.IsSpecified('expiration_date') or args.IsSpecified('retention_period')):
        raise c_exceptions.InvalidArgumentException('--expiration-date or --retention-period', 'Must specify either --expiration-date or --retention-period.')
    if args.expiration_date:
        return args.expiration_date
    elif args.retention_period:
        return ParseExpireTime(args.retention_period)