from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DeniedPermissionsValue(_messages.Message):
    """Lists all denied permissions in the deny rule and indicates whether
    each permission matches the permission in the request. Each key identifies
    a denied permission in the rule, and each value indicates whether the
    denied permission matches the permission in the request.

    Messages:
      AdditionalProperty: An additional property for a DeniedPermissionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        DeniedPermissionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DeniedPermissionsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3betaDenyRuleExplanationAn
          notatedPermissionMatching attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaDenyRuleExplanationAnnotatedPermissionMatching', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)