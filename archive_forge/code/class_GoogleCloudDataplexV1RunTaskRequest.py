from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1RunTaskRequest(_messages.Message):
    """A GoogleCloudDataplexV1RunTaskRequest object.

  Messages:
    ArgsValue: Optional. Execution spec arguments. If the map is left empty,
      the task will run with existing execution spec args from task
      definition. If the map contains an entry with a new key, the same will
      be added to existing set of args. If the map contains an entry with an
      existing arg key in task definition, the task will run with new arg
      value for that entry. Clearing an existing arg will require arg value to
      be explicitly set to a hyphen "-". The arg value cannot be empty.
    LabelsValue: Optional. User-defined labels for the task. If the map is
      left empty, the task will run with existing labels from task definition.
      If the map contains an entry with a new key, the same will be added to
      existing set of labels. If the map contains an entry with an existing
      label key in task definition, the task will run with new label value for
      that entry. Clearing an existing label will require label value to be
      explicitly set to a hyphen "-". The label value cannot be empty.

  Fields:
    args: Optional. Execution spec arguments. If the map is left empty, the
      task will run with existing execution spec args from task definition. If
      the map contains an entry with a new key, the same will be added to
      existing set of args. If the map contains an entry with an existing arg
      key in task definition, the task will run with new arg value for that
      entry. Clearing an existing arg will require arg value to be explicitly
      set to a hyphen "-". The arg value cannot be empty.
    labels: Optional. User-defined labels for the task. If the map is left
      empty, the task will run with existing labels from task definition. If
      the map contains an entry with a new key, the same will be added to
      existing set of labels. If the map contains an entry with an existing
      label key in task definition, the task will run with new label value for
      that entry. Clearing an existing label will require label value to be
      explicitly set to a hyphen "-". The label value cannot be empty.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ArgsValue(_messages.Message):
        """Optional. Execution spec arguments. If the map is left empty, the task
    will run with existing execution spec args from task definition. If the
    map contains an entry with a new key, the same will be added to existing
    set of args. If the map contains an entry with an existing arg key in task
    definition, the task will run with new arg value for that entry. Clearing
    an existing arg will require arg value to be explicitly set to a hyphen
    "-". The arg value cannot be empty.

    Messages:
      AdditionalProperty: An additional property for a ArgsValue object.

    Fields:
      additionalProperties: Additional properties of type ArgsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ArgsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the task. If the map is left empty,
    the task will run with existing labels from task definition. If the map
    contains an entry with a new key, the same will be added to existing set
    of labels. If the map contains an entry with an existing label key in task
    definition, the task will run with new label value for that entry.
    Clearing an existing label will require label value to be explicitly set
    to a hyphen "-". The label value cannot be empty.

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
    args = _messages.MessageField('ArgsValue', 1)
    labels = _messages.MessageField('LabelsValue', 2)