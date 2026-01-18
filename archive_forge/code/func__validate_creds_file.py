import os
import pprint
from twython import Twython
def _validate_creds_file(self, verbose=False):
    """Check validity of a credentials file."""
    oauth1 = False
    oauth1_keys = ['app_key', 'app_secret', 'oauth_token', 'oauth_token_secret']
    oauth2 = False
    oauth2_keys = ['app_key', 'app_secret', 'access_token']
    if all((k in self.oauth for k in oauth1_keys)):
        oauth1 = True
    elif all((k in self.oauth for k in oauth2_keys)):
        oauth2 = True
    if not (oauth1 or oauth2):
        msg = f'Missing or incorrect entries in {self.creds_file}\n'
        msg += pprint.pformat(self.oauth)
        raise ValueError(msg)
    elif verbose:
        print(f'Credentials file "{self.creds_file}" looks good')