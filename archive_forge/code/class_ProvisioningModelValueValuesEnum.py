from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvisioningModelValueValuesEnum(_messages.Enum):
    """Optional. Specifies the provisioning model of the instance.

    Values:
      PROVISIONING_MODEL_UNSPECIFIED: Default value. This value is not used.
      STANDARD: Standard provisioning with user controlled runtime, no
        discounts.
      SPOT: Heavily discounted, no guaranteed runtime.
    """
    PROVISIONING_MODEL_UNSPECIFIED = 0
    STANDARD = 1
    SPOT = 2