from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayGatewayConfig(_messages.Message):
    """Configuration settings for Gateways.

  Fields:
    backendConfig: Required. Backend settings that are applied to all backends
      of the Gateway.
  """
    backendConfig = _messages.MessageField('ApigatewayBackendConfig', 1)