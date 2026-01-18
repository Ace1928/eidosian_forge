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
def DeserializeMessage(self, response_type, data):
    """Deserialize the given data as method_config.response_type."""
    try:
        message = encoding.JsonToMessage(response_type, data)
    except (exceptions.InvalidDataFromServerError, messages.ValidationError, ValueError) as e:
        raise exceptions.InvalidDataFromServerError('Error decoding response "%s" as type %s: %s' % (data, response_type.__name__, e))
    return message