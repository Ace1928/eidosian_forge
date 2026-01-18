from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ViolationTypeValueValuesEnum(_messages.Enum):
    """Output only. Type of the violation

    Values:
      VIOLATION_TYPE_UNSPECIFIED: Unspecified type.
      ORG_POLICY: Org Policy Violation.
      RESOURCE: Resource Violation.
    """
    VIOLATION_TYPE_UNSPECIFIED = 0
    ORG_POLICY = 1
    RESOURCE = 2