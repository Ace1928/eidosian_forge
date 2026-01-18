import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def MapParamNames(params, request_type):
    """Reverse parameter remappings for URL construction."""
    return [encoding.GetCustomJsonFieldMapping(request_type, json_name=p) or p for p in params]