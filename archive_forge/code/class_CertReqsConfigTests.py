import os
import sys
from .. import config, tests, trace
from ..transport.http import opt_ssl_ca_certs, ssl
class CertReqsConfigTests(tests.TestCaseInTempDir):

    def test_default(self):
        stack = config.MemoryStack(b'')
        self.assertEqual(ssl.CERT_REQUIRED, stack.get('ssl.cert_reqs'))

    def test_from_string(self):
        stack = config.MemoryStack(b'ssl.cert_reqs = none\n')
        self.assertEqual(ssl.CERT_NONE, stack.get('ssl.cert_reqs'))
        stack = config.MemoryStack(b'ssl.cert_reqs = required\n')
        self.assertEqual(ssl.CERT_REQUIRED, stack.get('ssl.cert_reqs'))
        stack = config.MemoryStack(b'ssl.cert_reqs = invalid\n')
        self.assertRaises(config.ConfigOptionValueError, stack.get, 'ssl.cert_reqs')