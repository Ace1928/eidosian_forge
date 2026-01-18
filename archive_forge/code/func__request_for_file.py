import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _request_for_file(self, file_url):
    """Make call for file under provided URL."""
    response = self.session.get(file_url, stream=True)
    content_length = response.headers.get('content-length', None)
    if content_length is None:
        error_msg = 'Data from provided URL {url} is not supported. Lack of content-length Header in requested file response.'.format(url=file_url)
        raise FileNotSupportedError(error_msg)
    elif not content_length.isdigit():
        error_msg = 'Data from provided URL {url} is not supported. content-length header value is not a digit.'.format(url=file_url)
        raise FileNotSupportedError(error_msg)
    return response