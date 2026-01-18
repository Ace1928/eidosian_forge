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
def RegistriesDevicesListRequestHook(ref, args, req):
    """Add Api field query string mappings to list requests."""
    del ref
    del args
    msg = devices.GetMessagesModule()
    updated_requests_type = msg.CloudiotProjectsLocationsRegistriesDevicesListRequest
    for req_field, mapped_param in _CUSTOM_JSON_FIELD_MAPPINGS.items():
        encoding.AddCustomJsonFieldMapping(updated_requests_type, req_field, mapped_param)
    return req