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
def _RequireClassAttrs(obj, attrs):
    for attr in attrs:
        attr_name = attr.upper()
        if not hasattr(obj, '%s' % attr_name) or not getattr(obj, attr_name):
            msg = 'No %s specified for object of class %s.' % (attr_name, type(obj).__name__)
            raise exceptions.GeneratedClientError(msg)