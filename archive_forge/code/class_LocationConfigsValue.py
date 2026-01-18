from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LocationConfigsValue(_messages.Message):
    """Deployment configuration of the instance by locations (only regions
    are supported now). Map keys are regions in the string form.

    Messages:
      AdditionalProperty: An additional property for a LocationConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LocationConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LocationConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A LocationConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('LocationConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)