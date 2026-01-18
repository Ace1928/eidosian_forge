from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EmbeddedTask(_messages.Message):
    """EmbeddedTask defines a Task that is embedded in a Pipeline.

  Messages:
    AnnotationsValue: User annotations. See
      https://google.aip.dev/128#annotations

  Fields:
    annotations: User annotations. See https://google.aip.dev/128#annotations
    taskSpec: Spec to instantiate this TaskRun.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """User annotations. See https://google.aip.dev/128#annotations

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    taskSpec = _messages.MessageField('TaskSpec', 2)