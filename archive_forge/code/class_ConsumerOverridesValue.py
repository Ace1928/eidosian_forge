from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ConsumerOverridesValue(_messages.Message):
    """Quota overrides set by the consumer. Consumer overrides will only have
    an effect up to the max_limit specified in the service config, or the the
    producer override, if one exists.  The key for this map is one of the
    following:  - '<GROUP_NAME>/<LIMIT_NAME>' for quotas defined within quota
    groups, where GROUP_NAME is the google.api.QuotaGroup.name field and
    LIMIT_NAME is the google.api.QuotaLimit.name field from the service
    config.  For example: 'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for
    quotas defined without quota groups, where LIMIT_NAME is the
    google.api.QuotaLimit.name field from the service config. For example:
    'borrowedCountPerOrganization'.

    Messages:
      AdditionalProperty: An additional property for a ConsumerOverridesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ConsumerOverridesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ConsumerOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A QuotaLimitOverride attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('QuotaLimitOverride', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)