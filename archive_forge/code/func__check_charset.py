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
def _check_charset(self):
    if self.charset != 'UTF-8':
        raise DeprecationWarning("Requests are expected to be submitted in UTF-8, not %s. You can fix this by doing req = req.decode('%s')" % (self.charset, self.charset))