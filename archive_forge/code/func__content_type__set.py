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
def _content_type__set(self, value=None):
    if value is not None:
        value = str(value)
        if ';' not in value:
            content_type = self._content_type_raw
            if ';' in content_type:
                value += ';' + content_type.split(';', 1)[1]
    self._content_type_raw = value