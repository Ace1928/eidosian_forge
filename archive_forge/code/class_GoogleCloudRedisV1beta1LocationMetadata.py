from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRedisV1beta1LocationMetadata(_messages.Message):
    """This location metadata represents additional configuration options for a
  given location where a Redis instance may be created. All fields are output
  only. It is returned as content of the
  `google.cloud.location.Location.metadata` field.

  Messages:
    AvailableZonesValue: Output only. The set of available zones in the
      location. The map is keyed by the lowercase ID of each zone, as defined
      by GCE. These keys can be specified in `location_id` or
      `alternative_location_id` fields when creating a Redis instance.

  Fields:
    availableZones: Output only. The set of available zones in the location.
      The map is keyed by the lowercase ID of each zone, as defined by GCE.
      These keys can be specified in `location_id` or
      `alternative_location_id` fields when creating a Redis instance.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AvailableZonesValue(_messages.Message):
        """Output only. The set of available zones in the location. The map is
    keyed by the lowercase ID of each zone, as defined by GCE. These keys can
    be specified in `location_id` or `alternative_location_id` fields when
    creating a Redis instance.

    Messages:
      AdditionalProperty: An additional property for a AvailableZonesValue
        object.

    Fields:
      additionalProperties: Additional properties of type AvailableZonesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AvailableZonesValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudRedisV1beta1ZoneMetadata attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudRedisV1beta1ZoneMetadata', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    availableZones = _messages.MessageField('AvailableZonesValue', 1)