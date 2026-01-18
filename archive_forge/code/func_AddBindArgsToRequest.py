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
def AddBindArgsToRequest(ref, args, req):
    """Python hook for yaml gateways bind command to process resource_args."""
    del ref
    messages = devices.GetMessagesModule()
    gateway_ref = args.CONCEPTS.gateway.Parse()
    device_ref = args.CONCEPTS.device.Parse()
    registry_ref = gateway_ref.Parent()
    bind_request = messages.BindDeviceToGatewayRequest(deviceId=device_ref.Name(), gatewayId=gateway_ref.Name())
    req.bindDeviceToGatewayRequest = bind_request
    req.parent = registry_ref.RelativeName()
    return req