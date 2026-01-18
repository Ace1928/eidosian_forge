import requests
import testtools.matchers
from keystone.tests.functional import core as functests
class TestServerRunning(functests.BaseTestCase):

    def test_admin_responds_with_multiple_choices(self):
        resp = requests.get(self.ADMIN_URL)
        self.assertThat(resp.status_code, is_multiple_choices)

    def test_admin_versions(self):
        for version in versions:
            resp = requests.get(self.ADMIN_URL + '/' + version)
            self.assertThat(resp.status_code, testtools.matchers.Annotate('failed for version %s' % version, is_ok))

    def test_public_responds_with_multiple_choices(self):
        resp = requests.get(self.PUBLIC_URL)
        self.assertThat(resp.status_code, is_multiple_choices)

    def test_public_versions(self):
        for version in versions:
            resp = requests.get(self.PUBLIC_URL + '/' + version)
            self.assertThat(resp.status_code, testtools.matchers.Annotate('failed for version %s' % version, is_ok))

    def test_get_user_token(self):
        token = self.get_scoped_user_token()
        self.assertIsNotNone(token)

    def test_get_admin_token(self):
        token = self.get_scoped_admin_token()
        self.assertIsNotNone(token)