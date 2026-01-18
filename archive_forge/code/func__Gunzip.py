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
def _Gunzip(gzipped_content):
    """Returns gunzipped content from gzipped contents."""
    f = tempfile.NamedTemporaryFile(suffix='gz', mode='w+b', delete=False)
    try:
        f.write(gzipped_content)
        f.close()
        with gzip.open(f.name, 'rb') as h:
            decompressed_content = h.read()
        return decompressed_content
    finally:
        os.unlink(f.name)