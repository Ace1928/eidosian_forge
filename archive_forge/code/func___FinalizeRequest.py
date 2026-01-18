import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def __FinalizeRequest(self, http_request, url_builder):
    """Make any final general adjustments to the request."""
    if http_request.http_method == 'GET' and len(http_request.url) > _MAX_URL_LENGTH:
        http_request.http_method = 'POST'
        http_request.headers['x-http-method-override'] = 'GET'
        http_request.headers['content-type'] = 'application/x-www-form-urlencoded'
        http_request.body = url_builder.query
        url_builder.query_params = {}
    http_request.url = url_builder.url