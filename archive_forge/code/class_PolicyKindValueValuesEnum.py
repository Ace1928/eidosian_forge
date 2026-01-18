from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyKindValueValuesEnum(_messages.Enum):
    """Immutable. The kind of the policy to attach in this binding: + When
    the policy is empty, this field must be set. + When the policy is set,
    this field + can be left empty and will be set to the policy kind, or +
    must set to the input policy kind

    Values:
      POLICY_KIND_UNSPECIFIED: Unspecified policy kind; Not a valid state
      PRINCIPAL_ACCESS_BOUNDARY: Principal access boundary policy kind
    """
    POLICY_KIND_UNSPECIFIED = 0
    PRINCIPAL_ACCESS_BOUNDARY = 1