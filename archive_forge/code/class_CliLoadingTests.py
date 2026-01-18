import argparse
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from testtools import matchers
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class CliLoadingTests(utils.TestCase):

    def setUp(self):
        super(CliLoadingTests, self).setUp()
        self.parser = argparse.ArgumentParser()
        loading.register_session_argparse_arguments(self.parser)

    def get_session(self, val, **kwargs):
        args = self.parser.parse_args(val.split())
        return loading.load_session_from_argparse_arguments(args, **kwargs)

    def test_insecure_timeout(self):
        s = self.get_session('--insecure --timeout 5.5')
        self.assertFalse(s.verify)
        self.assertEqual(5.5, s.timeout)

    def test_client_certs(self):
        cert = '/path/to/certfile'
        key = '/path/to/keyfile'
        s = self.get_session('--os-cert %s --os-key %s' % (cert, key))
        self.assertTrue(s.verify)
        self.assertEqual((cert, key), s.cert)

    def test_cacert(self):
        cacert = '/path/to/cacert'
        s = self.get_session('--os-cacert %s' % cacert)
        self.assertEqual(cacert, s.verify)