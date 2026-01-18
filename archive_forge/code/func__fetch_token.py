from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
def _fetch_token(self, url, **request_kwargs):
    log.debug('Fetching token from %s using client %s', url, self._client.client)
    r = self.post(url, **request_kwargs)
    if r.status_code >= 400:
        error = "Token request failed with code %s, response was '%s'."
        raise TokenRequestDenied(error % (r.status_code, r.text), r)
    log.debug('Decoding token from response "%s"', r.text)
    try:
        token = dict(urldecode(r.text.strip()))
    except ValueError as e:
        error = 'Unable to decode token from token response. This is commonly caused by an unsuccessful request where a non urlencoded error message is returned. The decoding error was %s' % e
        raise ValueError(error)
    log.debug('Obtained token %s', token)
    log.debug('Updating internal client attributes from token data.')
    self._populate_attributes(token)
    self.token = token
    return token