import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
class TestOpenFunc:

    def setup_method(self):
        self.tmpdir = mkdtemp()

    def teardown_method(self):
        rmtree(self.tmpdir)

    def test_DataSourceOpen(self):
        local_file = valid_textfile(self.tmpdir)
        fp = datasource.open(local_file, destpath=self.tmpdir)
        assert_(fp)
        fp.close()
        fp = datasource.open(local_file)
        assert_(fp)
        fp.close()