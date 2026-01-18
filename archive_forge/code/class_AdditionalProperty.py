from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AdditionalProperty(_messages.Message):
    """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
    key = _messages.StringField(1)
    value = _messages.StringField(2)