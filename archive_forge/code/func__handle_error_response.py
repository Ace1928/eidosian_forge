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
def _handle_error_response(response_data, retryable_error):
    """Translates an error response into an exception.

    Args:
        response_data (Mapping | str): The decoded response data.
        retryable_error Optional[bool]: A boolean indicating if an error is retryable.
            Defaults to False.

    Raises:
        google.auth.exceptions.RefreshError: The errors contained in response_data.
    """
    retryable_error = retryable_error if retryable_error else False
    if isinstance(response_data, six.string_types):
        raise exceptions.RefreshError(response_data, retryable=retryable_error)
    try:
        error_details = '{}: {}'.format(response_data['error'], response_data.get('error_description'))
    except (KeyError, ValueError):
        error_details = json.dumps(response_data)
    raise exceptions.RefreshError(error_details, response_data, retryable=retryable_error)