from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class KubernetesServiceAccountAdmissionRulesValue(_messages.Message):
    """Optional. Per-kubernetes-service-account admission rules. Service
    account spec format: `namespace:serviceaccount`. e.g. `test-ns:default`

    Messages:
      AdditionalProperty: An additional property for a
        KubernetesServiceAccountAdmissionRulesValue object.

    Fields:
      additionalProperties: Additional properties of type
        KubernetesServiceAccountAdmissionRulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a
      KubernetesServiceAccountAdmissionRulesValue object.

      Fields:
        key: Name of the additional property.
        value: A AdmissionRule attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AdmissionRule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)