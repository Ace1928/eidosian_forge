from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CountsValue(_messages.Message):
    """Count of task in each state in the TaskGroup. The map key is task
    state name.

    Messages:
      AdditionalProperty: An additional property for a CountsValue object.

    Fields:
      additionalProperties: Additional properties of type CountsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CountsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)