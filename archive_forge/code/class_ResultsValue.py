from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ResultsValue(_messages.Message):
    """Map from filename to the final result for that file.

    Messages:
      AdditionalProperty: An additional property for a ResultsValue object.

    Fields:
      additionalProperties: Additional properties of type ResultsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ResultsValue object.

      Fields:
        key: Name of the additional property.
        value: A BatchRecognizeFileResult attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('BatchRecognizeFileResult', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)