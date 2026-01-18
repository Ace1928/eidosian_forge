import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def _text__set(self, value):
    if not self.charset:
        raise AttributeError('You cannot access Response.text unless charset is set')
    if not isinstance(value, text_type):
        raise TypeError('You can only set Request.text to a unicode string (not %s)' % type(value))
    self.body = value.encode(self.charset)