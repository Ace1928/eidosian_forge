from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.credentials import requests
from six.moves import http_client as httplib
class HttpRequestFailError(core_exceptions.Error):
    """Indicates that the http request fails in some way."""
    pass