from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMemcacheV1LocationMetadata(_messages.Message):
    """Metadata for the given google.cloud.location.Location.

  Messages:
    AvailableZonesValue: Output only. The set of available zones in the
      location. The map is keyed by the lowercase ID of each zone, as defined
      by GCE. These keys can be specified in the `zones` field when creating a
      Memcached instance.

  Fields:
    availableZones: Output only. The set of available zones in the location.
      The map is keyed by the lowercase ID of each zone, as defined by GCE.
      These keys can be specified in the `zones` field when creating a
      Memcached instance.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AvailableZonesValue(_messages.Message):
        """Output only. The set of available zones in the location. The map is
    keyed by the lowercase ID of each zone, as defined by GCE. These keys can
    be specified in the `zones` field when creating a Memcached instance.

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
        value: A GoogleCloudMemcacheV1ZoneMetadata attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudMemcacheV1ZoneMetadata', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    availableZones = _messages.MessageField('AvailableZonesValue', 1)