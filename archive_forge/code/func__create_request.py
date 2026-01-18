from __future__ import absolute_import, unicode_literals
import time
from oauthlib.common import Request, generate_token
from .. import (CONTENT_TYPE_FORM_URLENCODED, SIGNATURE_HMAC, SIGNATURE_RSA,
def _create_request(self, uri, http_method, body, headers):
    headers = headers or {}
    if 'Content-Type' in headers and CONTENT_TYPE_FORM_URLENCODED in headers['Content-Type']:
        request = Request(uri, http_method, body, headers)
    else:
        request = Request(uri, http_method, '', headers)
    signature_type, params, oauth_params = self._get_signature_type_and_params(request)
    if len(dict(oauth_params)) != len(oauth_params):
        raise errors.InvalidRequestError(description='Duplicate OAuth1 entries.')
    oauth_params = dict(oauth_params)
    request.signature = oauth_params.get('oauth_signature')
    request.client_key = oauth_params.get('oauth_consumer_key')
    request.resource_owner_key = oauth_params.get('oauth_token')
    request.nonce = oauth_params.get('oauth_nonce')
    request.timestamp = oauth_params.get('oauth_timestamp')
    request.redirect_uri = oauth_params.get('oauth_callback')
    request.verifier = oauth_params.get('oauth_verifier')
    request.signature_method = oauth_params.get('oauth_signature_method')
    request.realm = dict(params).get('realm')
    request.oauth_params = oauth_params
    request.params = [(k, v) for k, v in params if k != 'oauth_signature']
    if 'realm' in request.headers.get('Authorization', ''):
        request.params = [(k, v) for k, v in request.params if k != 'realm']
    return request