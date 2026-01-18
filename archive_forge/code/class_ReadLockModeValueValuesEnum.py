from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadLockModeValueValuesEnum(_messages.Enum):
    """Read lock mode for the transaction.

    Values:
      READ_LOCK_MODE_UNSPECIFIED: Default value. If the value is not
        specified, the pessimistic read lock is used.
      PESSIMISTIC: Pessimistic lock mode. Read locks are acquired immediately
        on read.
      OPTIMISTIC: Optimistic lock mode. Locks for reads within the transaction
        are not acquired on read. Instead the locks are acquired on a commit
        to validate that read/queried data has not changed since the
        transaction started.
    """
    READ_LOCK_MODE_UNSPECIFIED = 0
    PESSIMISTIC = 1
    OPTIMISTIC = 2