from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding as _encoding
from googlecloudsdk.core import exceptions
import six
def AddCustomJSONFieldMappingsToRequest(request_type, mappings):
    """Adds CustomJsonFieldMappings to the provided request_type.

  Args:
    request_type: (protorpc.messages.Message) request type for this API call
    mappings: (dict) Map from Python field names to JSON field names to be
      used on the wire.

  Returns:
    Updated request class containing the desired custom JSON mappings.
  """
    for req_field, mapped_param in mappings.items():
        _encoding.AddCustomJsonFieldMapping(message_type=request_type, python_name=req_field, json_name=mapped_param)
    return request_type