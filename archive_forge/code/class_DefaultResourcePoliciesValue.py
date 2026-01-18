from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DefaultResourcePoliciesValue(_messages.Message):
    """An optional parameter for storing the default resource policies that
    will be used for the Disks created in the given scope. The Key is a string
    type, provided by customers to uniquely identify the default Resource
    Policy entry. The Value is a Default ResourcePolicyDetails Object used to
    represent the detailed information of the Resource Policy entry.

    Messages:
      AdditionalProperty: An additional property for a
        DefaultResourcePoliciesValue object.

    Fields:
      additionalProperties: Additional properties of type
        DefaultResourcePoliciesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DefaultResourcePoliciesValue object.

      Fields:
        key: Name of the additional property.
        value: A DiskSettingsResourcePolicyDetails attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('DiskSettingsResourcePolicyDetails', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)