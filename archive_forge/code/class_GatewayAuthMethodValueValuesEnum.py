from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewayAuthMethodValueValuesEnum(_messages.Enum):
    """Indicates how to authorize and/or authenticate devices to access the
    gateway.

    Values:
      GATEWAY_AUTH_METHOD_UNSPECIFIED: No authentication/authorization method
        specified. No devices are allowed to access the gateway.
      ASSOCIATION_ONLY: The device is authenticated through the gateway
        association only. Device credentials are ignored even if provided.
      DEVICE_AUTH_TOKEN_ONLY: The device is authenticated through its own
        credentials. Gateway association is not checked.
      ASSOCIATION_AND_DEVICE_AUTH_TOKEN: The device is authenticated through
        both device credentials and gateway association. The device must be
        bound to the gateway and must provide its own credentials.
    """
    GATEWAY_AUTH_METHOD_UNSPECIFIED = 0
    ASSOCIATION_ONLY = 1
    DEVICE_AUTH_TOKEN_ONLY = 2
    ASSOCIATION_AND_DEVICE_AUTH_TOKEN = 3