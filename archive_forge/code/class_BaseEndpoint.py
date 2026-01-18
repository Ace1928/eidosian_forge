from __future__ import absolute_import, unicode_literals
import functools
import logging
from ..errors import (FatalClientError, OAuth2Error, ServerError,
class BaseEndpoint(object):

    def __init__(self):
        self._available = True
        self._catch_errors = False

    @property
    def available(self):
        return self._available

    @available.setter
    def available(self, available):
        self._available = available

    @property
    def catch_errors(self):
        return self._catch_errors

    @catch_errors.setter
    def catch_errors(self, catch_errors):
        self._catch_errors = catch_errors

    def _raise_on_missing_token(self, request):
        """Raise error on missing token."""
        if not request.token:
            raise InvalidRequestError(request=request, description='Missing token parameter.')

    def _raise_on_invalid_client(self, request):
        """Raise on failed client authentication."""
        if self.request_validator.client_authentication_required(request):
            if not self.request_validator.authenticate_client(request):
                log.debug('Client authentication failed, %r.', request)
                raise InvalidClientError(request=request)
        elif not self.request_validator.authenticate_client_id(request.client_id, request):
            log.debug('Client authentication failed, %r.', request)
            raise InvalidClientError(request=request)

    def _raise_on_unsupported_token(self, request):
        """Raise on unsupported tokens."""
        if request.token_type_hint and request.token_type_hint in self.valid_token_types and (request.token_type_hint not in self.supported_token_types):
            raise UnsupportedTokenTypeError(request=request)