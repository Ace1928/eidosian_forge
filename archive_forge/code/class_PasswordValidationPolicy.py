from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PasswordValidationPolicy(_messages.Message):
    """Database instance local user password validation policy

  Enums:
    ComplexityValueValuesEnum: The complexity of the password.

  Fields:
    complexity: The complexity of the password.
    disallowCompromisedCredentials: This field is deprecated and will be
      removed in a future version of the API.
    disallowUsernameSubstring: Disallow username as a part of the password.
    enablePasswordPolicy: Whether the password policy is enabled or not.
    minLength: Minimum number of characters allowed.
    passwordChangeInterval: Minimum interval after which the password can be
      changed. This flag is only supported for PostgreSQL.
    reuseInterval: Number of previous passwords that cannot be reused.
  """

    class ComplexityValueValuesEnum(_messages.Enum):
        """The complexity of the password.

    Values:
      COMPLEXITY_UNSPECIFIED: Complexity check is not specified.
      COMPLEXITY_DEFAULT: A combination of lowercase, uppercase, numeric, and
        non-alphanumeric characters.
    """
        COMPLEXITY_UNSPECIFIED = 0
        COMPLEXITY_DEFAULT = 1
    complexity = _messages.EnumField('ComplexityValueValuesEnum', 1)
    disallowCompromisedCredentials = _messages.BooleanField(2)
    disallowUsernameSubstring = _messages.BooleanField(3)
    enablePasswordPolicy = _messages.BooleanField(4)
    minLength = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    passwordChangeInterval = _messages.StringField(6)
    reuseInterval = _messages.IntegerField(7, variant=_messages.Variant.INT32)