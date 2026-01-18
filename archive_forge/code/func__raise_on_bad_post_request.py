import functools
import logging
from ..errors import (
def _raise_on_bad_post_request(self, request):
    """Raise if invalid POST request received
        """
    if request.http_method.upper() == 'POST':
        query_params = request.uri_query or ''
        if query_params:
            raise InvalidRequestError(request=request, description='URL query parameters are not allowed')