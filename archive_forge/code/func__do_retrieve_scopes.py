import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _do_retrieve_scopes(self, http, token):
    """Retrieves the list of authorized scopes from the OAuth2 provider.

        Args:
            http: an object to be used to make HTTP requests.
            token: A string used as the token to identify the credentials to
                   the provider.

        Raises:
            Error: When refresh fails, indicating the the access token is
                   invalid.
        """
    logger.info('Refreshing scopes')
    query_params = {'access_token': token, 'fields': 'scope'}
    token_info_uri = _helpers.update_query_params(self.token_info_uri, query_params)
    resp, content = transport.request(http, token_info_uri)
    content = _helpers._from_bytes(content)
    if resp.status == http_client.OK:
        d = json.loads(content)
        self.scopes = set(_helpers.string_to_scopes(d.get('scope', '')))
    else:
        error_msg = 'Invalid response {0}.'.format(resp.status)
        try:
            d = json.loads(content)
            if 'error_description' in d:
                error_msg = d['error_description']
        except (TypeError, ValueError):
            pass
        raise Error(error_msg)