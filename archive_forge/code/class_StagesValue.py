from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class StagesValue(_messages.Message):
    """A mapping from each stage to the information about that stage.

    Messages:
      AdditionalProperty: An additional property for a StagesValue object.

    Fields:
      additionalProperties: Additional properties of type StagesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a StagesValue object.

      Fields:
        key: Name of the additional property.
        value: A JobExecutionStageInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('JobExecutionStageInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)