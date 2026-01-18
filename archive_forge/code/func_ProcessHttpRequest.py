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
def ProcessHttpRequest(self, http_request):
    """Hook for pre-processing of http requests."""
    http_request.headers.update(self.additional_http_headers)
    if self.log_request:
        logging.info('Making http %s to %s', http_request.http_method, http_request.url)
        logging.info('Headers: %s', pprint.pformat(http_request.headers))
        if http_request.body:
            logging.info('Body:\n%s', http_request.loggable_body or http_request.body)
        else:
            logging.info('Body: (none)')
    return http_request