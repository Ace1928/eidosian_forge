from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpConfig(_messages.Message):
    """The configuration of the HTTP bridge for a device registry.

  Enums:
    HttpEnabledStateValueValuesEnum: If enabled, allows devices to use
      DeviceService via the HTTP protocol. Otherwise, any requests to
      DeviceService will fail for this registry.

  Fields:
    httpEnabledState: If enabled, allows devices to use DeviceService via the
      HTTP protocol. Otherwise, any requests to DeviceService will fail for
      this registry.
  """

    class HttpEnabledStateValueValuesEnum(_messages.Enum):
        """If enabled, allows devices to use DeviceService via the HTTP protocol.
    Otherwise, any requests to DeviceService will fail for this registry.

    Values:
      HTTP_STATE_UNSPECIFIED: No HTTP state specified. If not specified,
        DeviceService will be enabled by default.
      HTTP_ENABLED: Enables DeviceService (HTTP) service for the registry.
      HTTP_DISABLED: Disables DeviceService (HTTP) service for the registry.
    """
        HTTP_STATE_UNSPECIFIED = 0
        HTTP_ENABLED = 1
        HTTP_DISABLED = 2
    httpEnabledState = _messages.EnumField('HttpEnabledStateValueValuesEnum', 1)