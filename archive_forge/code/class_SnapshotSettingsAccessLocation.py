from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotSettingsAccessLocation(_messages.Message):
    """A SnapshotSettingsAccessLocation object.

  Messages:
    LocationsValue: List of regions that can restore a regional snapshot from
      the current region

  Fields:
    locations: List of regions that can restore a regional snapshot from the
      current region
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocationsValue(_messages.Message):
        """List of regions that can restore a regional snapshot from the current
    region

    Messages:
      AdditionalProperty: An additional property for a LocationsValue object.

    Fields:
      additionalProperties: Additional properties of type LocationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocationsValue object.

      Fields:
        key: Name of the additional property.
        value: A SnapshotSettingsAccessLocationAccessLocationPreference
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('SnapshotSettingsAccessLocationAccessLocationPreference', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    locations = _messages.MessageField('LocationsValue', 1)