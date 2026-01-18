from __future__ import absolute_import, unicode_literals
import functools
import logging
from ..errors import (FatalClientError, OAuth2Error, ServerError,
def _raise_on_invalid_client(self, request):
    """Raise on failed client authentication."""
    if self.request_validator.client_authentication_required(request):
        if not self.request_validator.authenticate_client(request):
            log.debug('Client authentication failed, %r.', request)
            raise InvalidClientError(request=request)
    elif not self.request_validator.authenticate_client_id(request.client_id, request):
        log.debug('Client authentication failed, %r.', request)
        raise InvalidClientError(request=request)