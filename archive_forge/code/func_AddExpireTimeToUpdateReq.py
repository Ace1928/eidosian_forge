from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.util import times
def AddExpireTimeToUpdateReq(unused_backup_ref, args, req):
    """Add expiration-date or retention-period to updateMask in the patch request."""
    req.backup.expireTime = GetExpireTime(args)
    req = AddFieldToUpdateMask('expire_time', req)
    return req