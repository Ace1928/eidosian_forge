from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MemberConfigsValue(_messages.Message):
    """Per-member configuration of workload certificate.

    Messages:
      AdditionalProperty: An additional property for a MemberConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MemberConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MemberConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A WorkloadCertificateMembershipSpec attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('WorkloadCertificateMembershipSpec', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)