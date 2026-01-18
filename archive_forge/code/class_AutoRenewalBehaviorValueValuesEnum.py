from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoRenewalBehaviorValueValuesEnum(_messages.Enum):
    """Optional. Auto renewal behavior of the subscription associated with
    the order.

    Values:
      AUTO_RENEWAL_BEHAVIOR_UNSPECIFIED: If unspecified, the auto renewal
        behavior will follow the default config.
      AUTO_RENEWAL_BEHAVIOR_ENABLE: Auto Renewal will be enabled on
        subscription.
      AUTO_RENEWAL_BEHAVIOR_DISABLE: Auto Renewal will be disabled on
        subscription.
    """
    AUTO_RENEWAL_BEHAVIOR_UNSPECIFIED = 0
    AUTO_RENEWAL_BEHAVIOR_ENABLE = 1
    AUTO_RENEWAL_BEHAVIOR_DISABLE = 2