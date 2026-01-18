from __future__ import absolute_import, unicode_literals
import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.uri_validate import is_absolute_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from ..request_validator import RequestValidator
def _handle_redirects(self, request):
    if request.redirect_uri is not None:
        request.using_default_redirect_uri = False
        log.debug('Using provided redirect_uri %s', request.redirect_uri)
        if not is_absolute_uri(request.redirect_uri):
            raise errors.InvalidRedirectURIError(request=request)
        if not self.request_validator.validate_redirect_uri(request.client_id, request.redirect_uri, request):
            raise errors.MismatchingRedirectURIError(request=request)
    else:
        request.redirect_uri = self.request_validator.get_default_redirect_uri(request.client_id, request)
        request.using_default_redirect_uri = True
        log.debug('Using default redirect_uri %s.', request.redirect_uri)
        if not request.redirect_uri:
            raise errors.MissingRedirectURIError(request=request)
        if not is_absolute_uri(request.redirect_uri):
            raise errors.InvalidRedirectURIError(request=request)