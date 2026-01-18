from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def AddCreateGatewayArgsToRequest(ref, args, req):
    """Python hook for yaml create command to process gateway flags."""
    del ref
    gateway = args.device_type
    auth_method = args.auth_method
    if not (gateway or auth_method):
        return req
    messages = devices.GetMessagesModule()
    req.device.gatewayConfig = messages.GatewayConfig()
    if auth_method:
        if not gateway or gateway == 'non-gateway':
            raise InvalidAuthMethodError('auth_method can only be set on gateway devices.')
        auth_enum = flags.GATEWAY_AUTH_METHOD_ENUM_MAPPER.GetEnumForChoice(auth_method)
        req.device.gatewayConfig.gatewayAuthMethod = auth_enum
    if gateway:
        gateway_enum = flags.CREATE_GATEWAY_ENUM_MAPPER.GetEnumForChoice(gateway)
        req.device.gatewayConfig.gatewayType = gateway_enum
    return req