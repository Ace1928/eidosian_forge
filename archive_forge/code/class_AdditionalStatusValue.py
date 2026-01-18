from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AdditionalStatusValue(_messages.Message):
    """Map to hold any additional status info for the operation If there is
    an accelerator being enabled/disabled/deleted, this will be populated with
    accelerator name as key and status as ENABLING, DISABLING or DELETING

    Messages:
      AdditionalProperty: An additional property for a AdditionalStatusValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AdditionalStatusValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AdditionalStatusValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)