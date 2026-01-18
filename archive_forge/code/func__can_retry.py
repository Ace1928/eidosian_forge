import datetime
import json
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _exponential_backoff
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
def _can_retry(status_code, response_data):
    """Checks if a request can be retried by inspecting the status code
    and response body of the request.

    Args:
        status_code (int): The response status code.
        response_data (Mapping | str): The decoded response data.

    Returns:
      bool: True if the response is retryable. False otherwise.
    """
    if status_code in transport.DEFAULT_RETRYABLE_STATUS_CODES:
        return True
    try:
        error_desc = response_data.get('error_description') or ''
        error_code = response_data.get('error') or ''
        if not isinstance(error_code, six.string_types) or not isinstance(error_desc, six.string_types):
            return False
        retryable_error_descriptions = {'internal_failure', 'server_error', 'temporarily_unavailable'}
        if any((e in retryable_error_descriptions for e in (error_code, error_desc))):
            return True
    except AttributeError:
        pass
    return False