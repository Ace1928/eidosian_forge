import logging
from oauthlib.common import generate_token, urldecode
from oauthlib.oauth2 import WebApplicationClient, InsecureTransportError
from oauthlib.oauth2 import LegacyApplicationClient
from oauthlib.oauth2 import TokenExpiredError, is_secure_transport
import requests
@access_token.deleter
def access_token(self):
    del self._client.access_token