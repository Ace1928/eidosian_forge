from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserTypeValueValuesEnum(_messages.Enum):
    """Optional. Type of this user.

    Values:
      USER_TYPE_UNSPECIFIED: Unspecified user type.
      ALLOYDB_BUILT_IN: The default user type that authenticates via password-
        based authentication.
      ALLOYDB_IAM_USER: Database user that can authenticate via IAM-Based
        authentication.
    """
    USER_TYPE_UNSPECIFIED = 0
    ALLOYDB_BUILT_IN = 1
    ALLOYDB_IAM_USER = 2