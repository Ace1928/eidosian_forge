from io import BytesIO
from .... import config, errors, osutils, tests
from .... import transport as _mod_transport
from ... import netrc_credential_store
class TestNetrcCS(tests.TestCaseInTempDir):

    def setUp(self):
        super().setUp()
        netrc_content = b'\nmachine host login joe password secret\ndefault login anonymous password joe@home\n'
        netrc_path = osutils.pathjoin(self.test_home_dir, '.netrc')
        with open(netrc_path, 'wb') as f:
            f.write(netrc_content)
        osutils.chmod_if_possible(netrc_path, 384)

    def _get_netrc_cs(self):
        return config.credential_store_registry.get_credential_store('netrc')

    def test_not_matching_user(self):
        cs = self._get_netrc_cs()
        password = cs.decode_password(dict(host='host', user='jim'))
        self.assertIs(None, password)

    def test_matching_user(self):
        cs = self._get_netrc_cs()
        password = cs.decode_password(dict(host='host', user='joe'))
        self.assertEqual('secret', password)

    def test_default_password(self):
        cs = self._get_netrc_cs()
        password = cs.decode_password(dict(host='other', user='anonymous'))
        self.assertEqual('joe@home', password)

    def test_default_password_without_user(self):
        cs = self._get_netrc_cs()
        password = cs.decode_password(dict(host='other'))
        self.assertIs(None, password)

    def test_get_netrc_credentials_via_auth_config(self):
        ac_content = b'\n[host1]\nhost = host\nuser = joe\npassword_encoding = netrc\n'
        conf = config.AuthenticationConfig(_file=BytesIO(ac_content))
        credentials = conf.get_credentials('scheme', 'host', user='joe')
        self.assertIsNot(None, credentials)
        self.assertEqual('secret', credentials.get('password', None))