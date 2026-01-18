import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def _initialize_headers(headers):
    """Creates a copy of the headers.

    Args:
        headers: dict, request headers to copy.

    Returns:
        dict, the copied headers or a new dictionary if the headers
        were None.
    """
    return {} if headers is None else dict(headers)