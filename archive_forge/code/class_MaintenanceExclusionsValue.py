from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MaintenanceExclusionsValue(_messages.Message):
    """Exceptions to maintenance window. Non-emergency maintenance should not
    occur in these windows.

    Messages:
      AdditionalProperty: An additional property for a
        MaintenanceExclusionsValue object.

    Fields:
      additionalProperties: Additional properties of type
        MaintenanceExclusionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MaintenanceExclusionsValue object.

      Fields:
        key: Name of the additional property.
        value: A TimeWindow attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('TimeWindow', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)