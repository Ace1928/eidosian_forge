from tests.compat import mock, unittest
from boto.pyami import config
from boto.compat import StringIO
class TestCanLoadConfigFile(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.file_contents = StringIO()
        file_contents = StringIO('[Boto]\nhttps_validate_certificates = true\nother = false\nhttp_socket_timeout = 1\n[Credentials]\naws_access_key_id=foo\naws_secret_access_key=bar\n')
        self.config = config.Config(fp=file_contents)

    def test_can_get_bool(self):
        self.assertTrue(self.config.getbool('Boto', 'https_validate_certificates'))
        self.assertFalse(self.config.getbool('Boto', 'other'))
        self.assertFalse(self.config.getbool('Boto', 'does-not-exist'))

    def test_can_get_int(self):
        self.assertEqual(self.config.getint('Boto', 'http_socket_timeout'), 1)
        self.assertEqual(self.config.getint('Boto', 'does-not-exist'), 0)
        self.assertEqual(self.config.getint('Boto', 'does-not-exist', default=20), 20)

    def test_can_get_strings(self):
        self.assertEqual(self.config.get('Credentials', 'aws_access_key_id'), 'foo')
        self.assertIsNone(self.config.get('Credentials', 'no-exist'))
        self.assertEqual(self.config.get('Credentials', 'no-exist', 'default-value'), 'default-value')