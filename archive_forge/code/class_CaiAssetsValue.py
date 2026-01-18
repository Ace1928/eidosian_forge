from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CaiAssetsValue(_messages.Message):
    """Output only. Map of Cloud Asset Inventory (CAI) type to CAI info (e.g.
    CAI ID). CAI type format follows https://cloud.google.com/asset-
    inventory/docs/supported-asset-types

    Messages:
      AdditionalProperty: An additional property for a CaiAssetsValue object.

    Fields:
      additionalProperties: Additional properties of type CaiAssetsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CaiAssetsValue object.

      Fields:
        key: Name of the additional property.
        value: A ResourceCAIInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ResourceCAIInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)