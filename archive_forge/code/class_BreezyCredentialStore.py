import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
class BreezyCredentialStore(CredentialStore):
    """Implementation of the launchpadlib CredentialStore API for Breezy.
    """

    def __init__(self, credential_save_failed=None):
        super().__init__(credential_save_failed)
        from breezy.config import AuthenticationConfig
        self.auth_config = AuthenticationConfig()

    def do_save(self, credentials, unique_key):
        """Store newly-authorized credentials in the keyring."""
        self.auth_config._set_option(unique_key, 'consumer_key', credentials.consumer.key)
        self.auth_config._set_option(unique_key, 'consumer_secret', credentials.consumer.secret)
        self.auth_config._set_option(unique_key, 'access_token', credentials.access_token.key)
        self.auth_config._set_option(unique_key, 'access_secret', credentials.access_token.secret)

    def do_load(self, unique_key):
        """Retrieve credentials from the keyring."""
        auth_def = self.auth_config._get_config().get(unique_key)
        if auth_def and auth_def.get('access_secret'):
            access_token = AccessToken(auth_def.get('access_token'), auth_def.get('access_secret'))
            return Credentials(consumer_name=auth_def.get('consumer_key'), consumer_secret=auth_def.get('consumer_secret'), access_token=access_token, application_name='Breezy')
        return None