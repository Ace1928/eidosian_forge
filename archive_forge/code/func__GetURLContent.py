from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
def _GetURLContent(url):
    """Download and return the content of URL."""
    response = urllib_request.urlopen(url)
    encoding = response.info().get('Content-Encoding')
    if encoding == 'gzip':
        content = _Gunzip(response.read())
    else:
        content = response.read()
    return content