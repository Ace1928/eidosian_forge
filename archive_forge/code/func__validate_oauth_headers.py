import uuid
from oauthlib import oauth1
from testtools import matchers
from keystoneauth1.extras import oauth1 as ksa_oauth1
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils as test_utils
def _validate_oauth_headers(self, auth_header, oauth_client):
    """Validate data in the headers.

        Assert that the data in the headers matches the data
        that is produced from oauthlib.
        """
    self.assertThat(auth_header, matchers.StartsWith('OAuth '))
    parameters = dict(oauth1.rfc5849.utils.parse_authorization_header(auth_header))
    self.assertEqual('HMAC-SHA1', parameters['oauth_signature_method'])
    self.assertEqual('1.0', parameters['oauth_version'])
    self.assertIsInstance(parameters['oauth_nonce'], str)
    self.assertEqual(oauth_client.client_key, parameters['oauth_consumer_key'])
    if oauth_client.resource_owner_key:
        self.assertEqual(oauth_client.resource_owner_key, parameters['oauth_token'])
    if oauth_client.verifier:
        self.assertEqual(oauth_client.verifier, parameters['oauth_verifier'])
    if oauth_client.callback_uri:
        self.assertEqual(oauth_client.callback_uri, parameters['oauth_callback'])
    return parameters