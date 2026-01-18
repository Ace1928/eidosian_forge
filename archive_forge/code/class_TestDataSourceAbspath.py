import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
class TestDataSourceAbspath:

    def setup_method(self):
        self.tmpdir = os.path.abspath(mkdtemp())
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        scheme, netloc, upath, pms, qry, frg = urlparse(valid_httpurl())
        local_path = os.path.join(self.tmpdir, netloc, upath.strip(os.sep).strip('/'))
        assert_equal(local_path, self.ds.abspath(valid_httpurl()))

    def test_ValidFile(self):
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        assert_equal(tmpfile, self.ds.abspath(tmpfilename))
        assert_equal(tmpfile, self.ds.abspath(tmpfile))

    def test_InvalidHTTP(self):
        scheme, netloc, upath, pms, qry, frg = urlparse(invalid_httpurl())
        invalidhttp = os.path.join(self.tmpdir, netloc, upath.strip(os.sep).strip('/'))
        assert_(invalidhttp != self.ds.abspath(valid_httpurl()))

    def test_InvalidFile(self):
        invalidfile = valid_textfile(self.tmpdir)
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        assert_(invalidfile != self.ds.abspath(tmpfilename))
        assert_(invalidfile != self.ds.abspath(tmpfile))

    def test_sandboxing(self):
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        tmp_path = lambda x: os.path.abspath(self.ds.abspath(x))
        assert_(tmp_path(valid_httpurl()).startswith(self.tmpdir))
        assert_(tmp_path(invalid_httpurl()).startswith(self.tmpdir))
        assert_(tmp_path(tmpfile).startswith(self.tmpdir))
        assert_(tmp_path(tmpfilename).startswith(self.tmpdir))
        for fn in malicious_files:
            assert_(tmp_path(http_path + fn).startswith(self.tmpdir))
            assert_(tmp_path(fn).startswith(self.tmpdir))

    def test_windows_os_sep(self):
        orig_os_sep = os.sep
        try:
            os.sep = '\\'
            self.test_ValidHTTP()
            self.test_ValidFile()
            self.test_InvalidHTTP()
            self.test_InvalidFile()
            self.test_sandboxing()
        finally:
            os.sep = orig_os_sep