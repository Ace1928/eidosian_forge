from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfidentialInstanceTypeValueValuesEnum(_messages.Enum):
    """Defines the type of technology used by the confidential instance.

    Values:
      CONFIDENTIAL_INSTANCE_TYPE_UNSPECIFIED: No type specified. Do not use
        this value.
      SEV: AMD Secure Encrypted Virtualization.
      SEV_SNP: AMD Secure Encrypted Virtualization - Secure Nested Paging.
      TDX: Intel Trust Domain eXtension.
    """
    CONFIDENTIAL_INSTANCE_TYPE_UNSPECIFIED = 0
    SEV = 1
    SEV_SNP = 2
    TDX = 3