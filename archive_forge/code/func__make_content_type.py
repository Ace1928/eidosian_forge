import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _make_content_type(mime_type=None):
    """Return ContentType for a given mime type.

    testtools was emitting a bad encoding, and this works around it.
    Unfortunately, is also loses data - probably want to drop this in a few
    releases.

    Based on mimeparse.py by Joe Gregorio (MIT License).

    https://github.com/conneg/mimeparse/blob/master/mimeparse.py
    """
    if mime_type is None:
        mime_type = 'application/octet-stream'
    msg = email.message.EmailMessage()
    msg['content-type'] = mime_type
    full_type, parameters = (msg.get_content_type(), dict(msg['content-type'].params))
    if full_type == '*':
        full_type = '*/*'
    type_parts = full_type.split('/') if '/' in full_type else None
    if not type_parts or len(type_parts) > 2:
        raise Exception("Can't parse type '%s'" % full_type)
    primary_type, sub_type = type_parts
    primary_type = primary_type.strip()
    sub_type = sub_type.strip()
    if 'charset' in parameters:
        if ',' in parameters['charset']:
            parameters['charset'] = parameters['charset'][:parameters['charset'].find(',')]
    return ContentType(primary_type, sub_type, parameters)