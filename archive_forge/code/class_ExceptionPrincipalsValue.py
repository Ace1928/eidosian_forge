from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ExceptionPrincipalsValue(_messages.Message):
    """Lists all exception principals in the deny rule and indicates whether
    each principal matches the principal in the request, either directly or
    through membership in a principal set. Each key identifies a exception
    principal in the rule, and each value indicates whether the exception
    principal matches the principal in the request.

    Messages:
      AdditionalProperty: An additional property for a
        ExceptionPrincipalsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ExceptionPrincipalsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ExceptionPrincipalsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3betaDenyRuleExplanationAn
          notatedDenyPrincipalMatching attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaDenyRuleExplanationAnnotatedDenyPrincipalMatching', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)