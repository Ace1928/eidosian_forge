from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class KubernetesNamespaceAdmissionRulesValue(_messages.Message):
    """Optional. Per-kubernetes-namespace admission rules. K8s namespace spec
    format: `[a-z.-]+`, e.g. `some-namespace`

    Messages:
      AdditionalProperty: An additional property for a
        KubernetesNamespaceAdmissionRulesValue object.

    Fields:
      additionalProperties: Additional properties of type
        KubernetesNamespaceAdmissionRulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a KubernetesNamespaceAdmissionRulesValue
      object.

      Fields:
        key: Name of the additional property.
        value: A AdmissionRule attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AdmissionRule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)