from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TransformNameMappingValue(_messages.Message):
    """Map of transform name prefixes of the job to be replaced to the
    corresponding name prefixes of the new job. Only applicable when updating
    a pipeline.

    Messages:
      AdditionalProperty: An additional property for a
        TransformNameMappingValue object.

    Fields:
      additionalProperties: Additional properties of type
        TransformNameMappingValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TransformNameMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)