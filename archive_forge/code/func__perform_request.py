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
def _perform_request():
    response = request(method='POST', url=token_uri, headers=headers, body=body, **kwargs)
    response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
    response_data = ''
    try:
        response_data = json.loads(response_body)
    except ValueError:
        response_data = response_body
    if response.status == http_client.OK:
        return (True, response_data, None)
    retryable_error = _can_retry(status_code=response.status, response_data=response_data)
    return (False, response_data, retryable_error)