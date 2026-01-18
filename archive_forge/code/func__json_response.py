import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def _json_response(response, result_key=None, error_message=None):
    """Temporary method to use to bridge from ShadeAdapter to SDK calls."""
    exceptions.raise_from_response(response, error_message=error_message)
    if not response.content:
        return response
    if 'application/json' not in response.headers.get('Content-Type'):
        return response
    try:
        result_json = response.json()
    except JSONDecodeError:
        return response
    return result_json