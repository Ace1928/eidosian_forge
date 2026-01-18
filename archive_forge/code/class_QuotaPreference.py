from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class QuotaPreference(_messages.Message):
    """QuotaPreference represents the preferred quota configuration specified
  for a project, folder or organization. There is only one QuotaPreference
  resource for a quota value targeting a unique set of dimensions.

  Messages:
    DimensionsValue: Immutable. The dimensions that this quota preference
      applies to. The key of the map entry is the name of a dimension, such as
      "region", "zone", "network_id", and the value of the map entry is the
      dimension value. If a dimension is missing from the map of dimensions,
      the quota preference applies to all the dimension values except for
      those that have other quota preferences configured for the specific
      value. NOTE: QuotaPreferences can only be applied across all values of
      "user" and "resource" dimension. Do not set values for "user" or
      "resource" in the dimension map. Example: {"provider", "Foo Inc"} where
      "provider" is a service specific dimension.

  Fields:
    contactEmail: Input only. An email address that can be used to contact the
      the user, in case Google Cloud needs more information to make a decision
      before additional quota can be granted. When requesting a quota
      increase, the email address is required. When requesting a quota
      decrease, the email address is optional. For example, the email address
      is optional when the `QuotaConfig.preferred_value` is smaller than the
      `QuotaDetails.reset_value`.
    createTime: Output only. Create time stamp
    dimensions: Immutable. The dimensions that this quota preference applies
      to. The key of the map entry is the name of a dimension, such as
      "region", "zone", "network_id", and the value of the map entry is the
      dimension value. If a dimension is missing from the map of dimensions,
      the quota preference applies to all the dimension values except for
      those that have other quota preferences configured for the specific
      value. NOTE: QuotaPreferences can only be applied across all values of
      "user" and "resource" dimension. Do not set values for "user" or
      "resource" in the dimension map. Example: {"provider", "Foo Inc"} where
      "provider" is a service specific dimension.
    etag: Optional. The current etag of the quota preference. If an etag is
      provided on update and does not match the current server's etag of the
      quota preference, the request will be blocked and an ABORTED error will
      be returned. See https://google.aip.dev/134#etags for more details on
      etags.
    justification: The reason / justification for this quota preference.
    name: Required except in the CREATE requests. The resource name of the
      quota preference. The ID component following "locations/" must be
      "global". Example: `projects/123/locations/global/quotaPreferences/my-
      config-for-us-east1`
    quotaConfig: Required. Preferred quota configuration.
    quotaId: Required. The id of the quota to which the quota preference is
      applied. A quota name is unique in the service. Example:
      `CpusPerProjectPerRegion`
    reconciling: Output only. Is the quota preference pending Google Cloud
      approval and fulfillment.
    service: Required. The name of the service to which the quota preference
      is applied.
    updateTime: Output only. Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionsValue(_messages.Message):
        """Immutable. The dimensions that this quota preference applies to. The
    key of the map entry is the name of a dimension, such as "region", "zone",
    "network_id", and the value of the map entry is the dimension value. If a
    dimension is missing from the map of dimensions, the quota preference
    applies to all the dimension values except for those that have other quota
    preferences configured for the specific value. NOTE: QuotaPreferences can
    only be applied across all values of "user" and "resource" dimension. Do
    not set values for "user" or "resource" in the dimension map. Example:
    {"provider", "Foo Inc"} where "provider" is a service specific dimension.

    Messages:
      AdditionalProperty: An additional property for a DimensionsValue object.

    Fields:
      additionalProperties: Additional properties of type DimensionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    contactEmail = _messages.StringField(1)
    createTime = _messages.StringField(2)
    dimensions = _messages.MessageField('DimensionsValue', 3)
    etag = _messages.StringField(4)
    justification = _messages.StringField(5)
    name = _messages.StringField(6)
    quotaConfig = _messages.MessageField('QuotaConfig', 7)
    quotaId = _messages.StringField(8)
    reconciling = _messages.BooleanField(9)
    service = _messages.StringField(10)
    updateTime = _messages.StringField(11)