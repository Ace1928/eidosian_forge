from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import requests
from six.moves import urllib
def _DecodeResponse(response):
    """Returns decoded string.

  Args:
    response: the raw string or bytes of JSON data

  Raises:
    ValueError: failure to load/decode JSON data
  """
    if not isinstance(response, str) and hasattr(response, 'decode'):
        response = response.decode()
    return response