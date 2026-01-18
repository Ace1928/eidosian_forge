from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchInstanceFilterGroupLabel(_messages.Message):
    """Represents a group of VMs that can be identified as having all these
  labels, for example "env=prod and app=web".

  Messages:
    LabelsValue: Compute Engine instance labels that must be present for a VM
      instance to be targeted by this filter.

  Fields:
    labels: Compute Engine instance labels that must be present for a VM
      instance to be targeted by this filter.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Compute Engine instance labels that must be present for a VM instance
    to be targeted by this filter.

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