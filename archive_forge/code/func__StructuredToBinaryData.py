from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core.util import times
def _StructuredToBinaryData(request_data_json):
    """Convert CloudEvents structured format to binary format.

  Args:
    request_data_json: dict, the parsed request body data

  Returns:
    cloudevent_data: str, the CloudEvent expected data with attributes in header
    cloudevent_headers: dict, the CloudEvent headers
  """
    cloudevent_headers = {}
    cloudevent_data = None
    for key, value in list(request_data_json.items()):
        normalized_key = key.lower()
        if normalized_key == 'data':
            cloudevent_data = value
        elif normalized_key in _FIELDS:
            cloudevent_headers['ce-' + normalized_key] = value
        elif normalized_key == 'datacontenttype':
            cloudevent_headers[_CONTENT_TYPE] = value
        else:
            cloudevent_headers[normalized_key] = value
    if _CONTENT_TYPE not in cloudevent_headers:
        cloudevent_headers[_CONTENT_TYPE] = 'application/json'
    return (json.dumps(cloudevent_data), cloudevent_headers)