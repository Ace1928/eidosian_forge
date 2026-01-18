from __future__ import absolute_import, unicode_literals
import functools
import logging
from ..errors import (FatalClientError, OAuth2Error, ServerError,
def _raise_on_missing_token(self, request):
    """Raise error on missing token."""
    if not request.token:
        raise InvalidRequestError(request=request, description='Missing token parameter.')