import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from oauthlib.uri_validate import is_absolute_uri
from ..request_validator import RequestValidator
from ..utils import is_secure_transport
def _create_cors_headers(self, request):
    """If CORS is allowed, create the appropriate headers."""
    if 'origin' not in request.headers:
        return {}
    origin = request.headers['origin']
    if not is_secure_transport(origin):
        log.debug('Origin "%s" is not HTTPS, CORS not allowed.', origin)
        return {}
    elif not self.request_validator.is_origin_allowed(request.client_id, origin, request):
        log.debug('Invalid origin "%s", CORS not allowed.', origin)
        return {}
    else:
        log.debug('Valid origin "%s", injecting CORS headers.', origin)
        return {'Access-Control-Allow-Origin': origin}