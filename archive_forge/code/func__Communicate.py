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
def _Communicate(url, method, body, headers):
    """Returns HTTP status, reason, and response body for a given HTTP request."""
    response = requests.GetSession().request(method, url, data=body, headers=headers, stream=True)
    status = response.status_code
    reason = response.reason
    data = response.content
    return (status, reason, data)