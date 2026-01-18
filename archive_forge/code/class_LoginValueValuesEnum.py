from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoginValueValuesEnum(_messages.Enum):
    """Level of login required to access this resource. Not supported for
    Node.js in the App Engine standard environment.

    Values:
      LOGIN_UNSPECIFIED: Not specified. LOGIN_OPTIONAL is assumed.
      LOGIN_OPTIONAL: Does not require that the user is signed in.
      LOGIN_ADMIN: If the user is not signed in, the auth_fail_action is
        taken. In addition, if the user is not an administrator for the
        application, they are given an error message regardless of
        auth_fail_action. If the user is an administrator, the handler
        proceeds.
      LOGIN_REQUIRED: If the user has signed in, the handler proceeds
        normally. Otherwise, the auth_fail_action is taken.
    """
    LOGIN_UNSPECIFIED = 0
    LOGIN_OPTIONAL = 1
    LOGIN_ADMIN = 2
    LOGIN_REQUIRED = 3