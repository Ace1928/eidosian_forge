from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def _GetGatewayEnum(parent='list_request'):
    """Get GatewayTypeValueEnum from the specified parent message."""
    messages = apis.GetMessagesModule('cloudiot', 'v1')
    if parent == 'list_request':
        request = messages.CloudiotProjectsLocationsRegistriesDevicesListRequest
    else:
        request = messages.GatewayConfig
    return request.GatewayTypeValueValuesEnum