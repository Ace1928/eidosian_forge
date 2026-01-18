import functools
import logging
from ..errors import (
def _raise_on_bad_method(self, request):
    if self.valid_request_methods is None:
        raise ValueError('Configure "valid_request_methods" property first')
    if request.http_method.upper() not in self.valid_request_methods:
        raise InvalidRequestError(request=request, description='Unsupported request method %s' % request.http_method.upper())