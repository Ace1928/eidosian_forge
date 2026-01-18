from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleRedirectOptions(_messages.Message):
    """A SecurityPolicyRuleRedirectOptions object.

  Enums:
    TypeValueValuesEnum: Type of the redirect action.

  Fields:
    target: Target for the redirect action. This is required if the type is
      EXTERNAL_302 and cannot be specified for GOOGLE_RECAPTCHA.
    type: Type of the redirect action.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of the redirect action.

    Values:
      EXTERNAL_302: <no description>
      GOOGLE_RECAPTCHA: <no description>
    """
        EXTERNAL_302 = 0
        GOOGLE_RECAPTCHA = 1
    target = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)