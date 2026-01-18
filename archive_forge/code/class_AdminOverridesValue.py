from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AdminOverridesValue(_messages.Message):
    """Quota overrides set by an administrator of a consumer organization or
    folder. The administrator of an organization or folder can set admin
    overrides for any folders or projects beneath it. When a project or folder
    moves out of the folder or organization that sets the admin override, the
    admin override will be removed. The key for this map is the same as the
    key for consumer_overrides.

    Messages:
      AdditionalProperty: An additional property for a AdminOverridesValue
        object.

    Fields:
      additionalProperties: Additional properties of type AdminOverridesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AdminOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A QuotaLimitOverride attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('QuotaLimitOverride', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)