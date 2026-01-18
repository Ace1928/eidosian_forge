from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidatorValueValuesEnum(_messages.Enum):
    """Validator type to validate membership with.

    Values:
      VALIDATOR_TYPE_UNSPECIFIED: UNSPECIFIED validator.
      MEMBERSHIP_ID: MEMBERSHIP_ID validator validates that the membership_id
        is still available.
      CROSS_PROJECT_PERMISSION: CROSS_PROJECT_PERMISSION validator validates
        that the cross-project role binding for the service agent is in place.
    """
    VALIDATOR_TYPE_UNSPECIFIED = 0
    MEMBERSHIP_ID = 1
    CROSS_PROJECT_PERMISSION = 2