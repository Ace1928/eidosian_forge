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
def _content_type__get(self):
    """Return the content type, but leaving off any parameters (like
        charset, but also things like the type in ``application/atom+xml;
        type=entry``)

        If you set this property, you can include parameters, or if
        you don't include any parameters in the value then existing
        parameters will be preserved.
        """
    return self._content_type_raw.split(';', 1)[0]