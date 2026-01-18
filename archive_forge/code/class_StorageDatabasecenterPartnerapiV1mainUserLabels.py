from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainUserLabels(_messages.Message):
    """Message type for storing user labels. User labels are used to tag App
  Engine resources, allowing users to search for resources matching a set of
  labels and to aggregate usage data by labels.

  Messages:
    LabelsValue: A LabelsValue object.

  Fields:
    labels: A LabelsValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """A LabelsValue object.

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