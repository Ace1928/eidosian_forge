from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.util import times
def GetExpireTime(args):
    """Parse flags for expire time."""
    if args.expiration_date:
        return args.expiration_date
    elif args.retention_period:
        return ParseExpireTime(args.retention_period)