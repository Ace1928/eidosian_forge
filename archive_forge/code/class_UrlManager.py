from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import select
import socket
import sys
import webbrowser
import wsgiref
from google_auth_oauthlib import flow as google_auth_flow
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as c_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import pkg_resources
from oauthlib.oauth2.rfc6749 import errors as rfc6749_errors
from requests import exceptions as requests_exceptions
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves.urllib import parse
class UrlManager(object):
    """A helper for url manipulation."""

    def __init__(self, url):
        self._parse_url = parse.urlparse(url)
        self._scheme, self._netloc, self._path, self._query = (self._parse_url.scheme, self._parse_url.netloc, self._parse_url.path, self._parse_url.query)
        self._parsed_query = parse.parse_qsl(self._query)

    def UpdateQueryParams(self, query_params):
        """Updates query params in the url using query_params.

    Args:
       query_params: A list of two-element tuples. The first element in the
         tuple is the query key and the second element is the query value.
    """
        for key, value in query_params:
            self._RemoveQueryParam(key)
            self._parsed_query.append((key, value))

    def RemoveQueryParams(self, query_keys):
        """Removes query params from the url.

    Args:
      query_keys: A list of query keys to remove.
    """
        for p in query_keys:
            self._RemoveQueryParam(p)

    def _RemoveQueryParam(self, query_key):
        self._parsed_query[:] = [p for p in self._parsed_query if p[0] != query_key]

    def ContainQueryParams(self, query_keys):
        """If the url contains the query keys in query_key.

    Args:
      query_keys: A list of query keys to check in the url.

    Returns:
      True if all query keys in query_keys are contained in url. Otherwise,
        return False.
    """
        parsed_query_keys = {k for k, v in self._parsed_query}
        return all([p in parsed_query_keys for p in query_keys])

    def GetQueryParam(self, query_key):
        """Gets the value of the query_key.

    Args:
       query_key: str, A query key to get the value for.

    Returns:
      The value of the query_key. None if query_key does not exist in the url.
    """
        for k, v in self._parsed_query:
            if query_key == k:
                return v

    def GetUrl(self):
        """Gets the current url in the string format."""
        encoded_query = parse.urlencode(self._parsed_query)
        return parse.urlunparse((self._scheme, self._netloc, self._path, '', encoded_query, ''))

    def GetPort(self):
        try:
            _, port = self._netloc.rsplit(':', 1)
            return int(port)
        except ValueError:
            return None