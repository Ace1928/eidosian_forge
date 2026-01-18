import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
def _Gzip(raw_content):
    """Returns gzipped content from any content."""
    f = tempfile.NamedTemporaryFile(suffix='gz', mode='wb', delete=False)
    f.close()
    try:
        with gzip.open(f.name, 'wb') as h:
            h.write(raw_content)
        with open(f.name, 'rb') as h:
            return h.read()
    finally:
        os.unlink(f.name)