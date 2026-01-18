from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def RetentionPolicyToString(retention_policy, bucket_url):
    """Converts Retention Policy to Human readable format."""
    retention_policy_str = ''
    if retention_policy and retention_policy.retentionPeriod:
        locked_string = '(LOCKED)' if retention_policy.isLocked else '(UNLOCKED)'
        retention_period = _RetentionPeriodToString(retention_policy.retentionPeriod)
        retention_effective_time = '    Effective Time: {}'.format(retention_policy.effectiveTime.strftime('%a, %d %b %Y %H:%M:%S GMT'))
        retention_policy_str = '  Retention Policy {}:\n{}\n{}'.format(locked_string, retention_period, retention_effective_time)
    else:
        retention_policy_str = '{} has no Retention Policy.'.format(bucket_url)
    return retention_policy_str