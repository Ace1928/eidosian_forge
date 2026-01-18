from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExpireBehaviorValueValuesEnum(_messages.Enum):
    """Specifies the expiration behavior of a free instance. The default of
    ExpireBehavior is `REMOVE_AFTER_GRACE_PERIOD`. This can be modified during
    or after creation, and before expiration.

    Values:
      EXPIRE_BEHAVIOR_UNSPECIFIED: Not specified.
      FREE_TO_PROVISIONED: When the free instance expires, upgrade the
        instance to a provisioned instance.
      REMOVE_AFTER_GRACE_PERIOD: When the free instance expires, disable the
        instance, and delete it after the grace period passes if it has not
        been upgraded.
    """
    EXPIRE_BEHAVIOR_UNSPECIFIED = 0
    FREE_TO_PROVISIONED = 1
    REMOVE_AFTER_GRACE_PERIOD = 2