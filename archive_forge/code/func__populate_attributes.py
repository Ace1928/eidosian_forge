from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
def _populate_attributes(self, token):
    if 'oauth_token' in token:
        self._client.client.resource_owner_key = token['oauth_token']
    else:
        raise TokenMissing('Response does not contain a token: {resp}'.format(resp=token), token)
    if 'oauth_token_secret' in token:
        self._client.client.resource_owner_secret = token['oauth_token_secret']
    if 'oauth_verifier' in token:
        self._client.client.verifier = token['oauth_verifier']