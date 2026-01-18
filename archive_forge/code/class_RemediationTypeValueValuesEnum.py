from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemediationTypeValueValuesEnum(_messages.Enum):
    """The type of remediation that can be applied.

    Values:
      REMEDIATION_TYPE_UNSPECIFIED: No remediation type specified.
      MITIGATION: A MITIGATION is available.
      NO_FIX_PLANNED: No fix is planned.
      NONE_AVAILABLE: Not available.
      VENDOR_FIX: A vendor fix is available.
      WORKAROUND: A workaround is available.
    """
    REMEDIATION_TYPE_UNSPECIFIED = 0
    MITIGATION = 1
    NO_FIX_PLANNED = 2
    NONE_AVAILABLE = 3
    VENDOR_FIX = 4
    WORKAROUND = 5