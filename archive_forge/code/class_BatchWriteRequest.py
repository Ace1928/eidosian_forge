from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchWriteRequest(_messages.Message):
    """The request for Firestore.BatchWrite.

  Messages:
    LabelsValue: Labels associated with this batch write.

  Fields:
    labels: Labels associated with this batch write.
    writes: The writes to apply. Method does not apply writes atomically and
      does not guarantee ordering. Each write succeeds or fails independently.
      You cannot write to the same document more than once per request.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with this batch write.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    writes = _messages.MessageField('Write', 2, repeated=True)