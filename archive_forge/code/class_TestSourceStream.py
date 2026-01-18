import gzip
import os
import re
import tempfile
from .... import tests
from ....tests import features
from ....tests.blackbox import ExternalBase
from ..cmds import _get_source_stream
from . import FastimportFeature
from :1
from :2
from :1
from :2
class TestSourceStream(tests.TestCase):
    _test_needs_features = [FastimportFeature]

    def test_get_source_stream_stdin(self):
        self.assertIsNot(None, _get_source_stream('-'))

    def test_get_source_gz(self):
        fd, filename = tempfile.mkstemp(suffix='.gz')
        with gzip.GzipFile(fileobj=os.fdopen(fd, 'wb'), mode='wb') as f:
            f.write(b'bla')
        stream = _get_source_stream(filename)
        self.assertIsNot('bla', stream.read())

    def test_get_source_file(self):
        fd, filename = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as f:
            f.write(b'bla')
        stream = _get_source_stream(filename)
        self.assertIsNot(b'bla', stream.read())