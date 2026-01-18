from keystoneauth1 import exceptions as ksa_exceptions
import testresources
from testtools import matchers
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils as test_utils
from keystoneclient import utils
class HashSignedTokenTestCase(test_utils.TestCase, testresources.ResourcedTestCase):
    """Unit tests for utils.hash_signed_token()."""
    resources = [('examples', client_fixtures.EXAMPLES_RESOURCE)]

    def test_default_md5(self):
        """The default hash method is md5."""
        token = self.examples.SIGNED_TOKEN_SCOPED
        token = token.encode('utf-8')
        token_id_default = utils.hash_signed_token(token)
        token_id_md5 = utils.hash_signed_token(token, mode='md5')
        self.assertThat(token_id_default, matchers.Equals(token_id_md5))
        self.assertThat(token_id_default, matchers.HasLength(32))

    def test_sha256(self):
        """Can also hash with sha256."""
        token = self.examples.SIGNED_TOKEN_SCOPED
        token = token.encode('utf-8')
        token_id = utils.hash_signed_token(token, mode='sha256')
        self.assertThat(token_id, matchers.HasLength(64))