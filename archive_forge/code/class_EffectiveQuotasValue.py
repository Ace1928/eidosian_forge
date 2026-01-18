from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EffectiveQuotasValue(_messages.Message):
    """The effective quota limits for each group, derived from the service
    defaults together with any producer or consumer overrides. For each limit,
    the effective value is the minimum of the producer and consumer overrides
    if either is present, or else the service default if neither is present.
    DEPRECATED. Use effective_quota_groups instead.

    Messages:
      AdditionalProperty: An additional property for a EffectiveQuotasValue
        object.

    Fields:
      additionalProperties: Additional properties of type EffectiveQuotasValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EffectiveQuotasValue object.

      Fields:
        key: Name of the additional property.
        value: A QuotaLimitOverride attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('QuotaLimitOverride', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)