from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PartnerMetadataValue(_messages.Message):
    """Partner Metadata assigned to the instance. A map from a subdomain to
    entries map. Subdomain name must be compliant with RFC1035 definition. The
    total size of all keys and values must be less than 2MB. Subdomain
    'metadata.compute.googleapis.com' is reserverd for instance's metadata.

    Messages:
      AdditionalProperty: An additional property for a PartnerMetadataValue
        object.

    Fields:
      additionalProperties: Additional properties of type PartnerMetadataValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PartnerMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A StructuredEntries attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('StructuredEntries', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)