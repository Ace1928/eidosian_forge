from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesListRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesDevicesListRequest object.

  Enums:
    GatewayListOptionsGatewayTypeValueValuesEnum: If `GATEWAY` is specified,
      only gateways are returned. If `NON_GATEWAY` is specified, only non-
      gateway devices are returned. If `GATEWAY_TYPE_UNSPECIFIED` is
      specified, all devices are returned.

  Fields:
    deviceIds: A list of device string IDs. For example, `['device0',
      'device12']`. If empty, this field is ignored. Maximum IDs: 10,000
    deviceNumIds: A list of device numeric IDs. If empty, this field is
      ignored. Maximum IDs: 10,000.
    fieldMask: The fields of the `Device` resource to be returned in the
      response. The fields `id` and `num_id` are always returned, along with
      any other fields specified in snake_case format, for example:
      `last_heartbeat_time`.
    gatewayListOptions_associationsDeviceId: If set, returns only the gateways
      with which the specified device is associated. The device ID can be
      numeric (`num_id`) or the user-defined string (`id`). For example, if
      `456` is specified, returns only the gateways to which the device with
      `num_id` 456 is bound.
    gatewayListOptions_associationsGatewayId: If set, only devices associated
      with the specified gateway are returned. The gateway ID can be numeric
      (`num_id`) or the user-defined string (`id`). For example, if `123` is
      specified, only devices bound to the gateway with `num_id` 123 are
      returned.
    gatewayListOptions_gatewayType: If `GATEWAY` is specified, only gateways
      are returned. If `NON_GATEWAY` is specified, only non-gateway devices
      are returned. If `GATEWAY_TYPE_UNSPECIFIED` is specified, all devices
      are returned.
    pageSize: The maximum number of devices to return in the response. If this
      value is zero, the service will select a default size. A call may return
      fewer objects than requested. A non-empty `next_page_token` in the
      response indicates that more data is available.
    pageToken: The value returned by the last `ListDevicesResponse`; indicates
      that this is a continuation of a prior `ListDevices` call and the system
      should return the next page of data.
    parent: Required. The device registry path. Required. For example,
      `projects/my-project/locations/us-central1/registries/my-registry`.
  """

    class GatewayListOptionsGatewayTypeValueValuesEnum(_messages.Enum):
        """If `GATEWAY` is specified, only gateways are returned. If
    `NON_GATEWAY` is specified, only non-gateway devices are returned. If
    `GATEWAY_TYPE_UNSPECIFIED` is specified, all devices are returned.

    Values:
      GATEWAY_TYPE_UNSPECIFIED: If unspecified, the device is considered a
        non-gateway device.
      GATEWAY: The device is a gateway.
      NON_GATEWAY: The device is not a gateway.
    """
        GATEWAY_TYPE_UNSPECIFIED = 0
        GATEWAY = 1
        NON_GATEWAY = 2
    deviceIds = _messages.StringField(1, repeated=True)
    deviceNumIds = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.UINT64)
    fieldMask = _messages.StringField(3)
    gatewayListOptions_associationsDeviceId = _messages.StringField(4)
    gatewayListOptions_associationsGatewayId = _messages.StringField(5)
    gatewayListOptions_gatewayType = _messages.EnumField('GatewayListOptionsGatewayTypeValueValuesEnum', 6)
    pageSize = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(8)
    parent = _messages.StringField(9, required=True)