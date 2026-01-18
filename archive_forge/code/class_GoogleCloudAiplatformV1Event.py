from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Event(_messages.Message):
    """An edge describing the relationship between an Artifact and an Execution
  in a lineage graph.

  Enums:
    TypeValueValuesEnum: Required. The type of the Event.

  Messages:
    LabelsValue: The labels with user-defined metadata to annotate Events.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one Event (System labels are
      excluded). See https://goo.gl/xmQnxf for more information and examples
      of labels. System reserved label keys are prefixed with
      "aiplatform.googleapis.com/" and are immutable.

  Fields:
    artifact: Required. The relative resource name of the Artifact in the
      Event.
    eventTime: Output only. Time the Event occurred.
    execution: Output only. The relative resource name of the Execution in the
      Event.
    labels: The labels with user-defined metadata to annotate Events. Label
      keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one Event (System labels are
      excluded). See https://goo.gl/xmQnxf for more information and examples
      of labels. System reserved label keys are prefixed with
      "aiplatform.googleapis.com/" and are immutable.
    type: Required. The type of the Event.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the Event.

    Values:
      TYPE_UNSPECIFIED: Unspecified whether input or output of the Execution.
      INPUT: An input of the Execution.
      OUTPUT: An output of the Execution.
    """
        TYPE_UNSPECIFIED = 0
        INPUT = 1
        OUTPUT = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to annotate Events. Label keys
    and values can be no longer than 64 characters (Unicode codepoints), can
    only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. No more than 64 user labels
    can be associated with one Event (System labels are excluded). See
    https://goo.gl/xmQnxf for more information and examples of labels. System
    reserved label keys are prefixed with "aiplatform.googleapis.com/" and are
    immutable.

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
    artifact = _messages.StringField(1)
    eventTime = _messages.StringField(2)
    execution = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)