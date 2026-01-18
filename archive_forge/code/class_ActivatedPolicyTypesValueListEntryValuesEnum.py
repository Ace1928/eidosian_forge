from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActivatedPolicyTypesValueListEntryValuesEnum(_messages.Enum):
    """ActivatedPolicyTypesValueListEntryValuesEnum enum type.

    Values:
      POLICY_TYPE_UNSPECIFIED: Unspecified policy type.
      FINE_GRAINED_ACCESS_CONTROL: Fine-grained access control policy that
        enables access control on tagged sub-resources.
    """
    POLICY_TYPE_UNSPECIFIED = 0
    FINE_GRAINED_ACCESS_CONTROL = 1