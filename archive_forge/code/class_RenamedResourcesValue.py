from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class RenamedResourcesValue(_messages.Message):
    """Map from full resource types to the effective short name for the
    resource. This is used when otherwise resource named from different
    services would cause naming collisions. Example entry:
    "datalabeling.googleapis.com/Dataset": "DataLabelingDataset"

    Messages:
      AdditionalProperty: An additional property for a RenamedResourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        RenamedResourcesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a RenamedResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)