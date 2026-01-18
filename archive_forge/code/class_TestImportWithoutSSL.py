import sys
import pytest
class TestImportWithoutSSL(TestWithoutSSL):

    def test_cannot_import_ssl(self):
        with pytest.raises(ImportError):
            import ssl

    def test_import_urllib3(self):
        import urllib3