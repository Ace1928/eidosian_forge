from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class KnownTokens:
    """Known access token/secret combinations."""

    def __init__(self, token_string, access_secret):
        self.token_string = token_string
        self.access_secret = access_secret
        self.token = AccessToken(token_string, access_secret)
        self.credentials = Credentials(consumer_name='launchpad-library', access_token=self.token)

    def login(self, cache=None, timeout=None, proxy_info=None, version=Launchpad.DEFAULT_VERSION):
        """Create a Launchpad object using these credentials."""
        return TestableLaunchpad(self.credentials, cache=cache, timeout=timeout, proxy_info=proxy_info, version=version)