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
def _do_refresh_request(self, http):
    """Refresh the access_token using the refresh_token.

        Args:
            http: an object to be used to make HTTP requests.

        Raises:
            HttpAccessTokenRefreshError: When the refresh fails.
        """
    body = self._generate_refresh_request_body()
    headers = self._generate_refresh_request_headers()
    logger.info('Refreshing access_token')
    resp, content = transport.request(http, self.token_uri, method='POST', body=body, headers=headers)
    content = _helpers._from_bytes(content)
    if resp.status == http_client.OK:
        d = json.loads(content)
        self.token_response = d
        self.access_token = d['access_token']
        self.refresh_token = d.get('refresh_token', self.refresh_token)
        if 'expires_in' in d:
            delta = datetime.timedelta(seconds=int(d['expires_in']))
            self.token_expiry = delta + _UTCNOW()
        else:
            self.token_expiry = None
        if 'id_token' in d:
            self.id_token = _extract_id_token(d['id_token'])
            self.id_token_jwt = d['id_token']
        else:
            self.id_token = None
            self.id_token_jwt = None
        self.invalid = False
        if self.store:
            self.store.locked_put(self)
    else:
        logger.info('Failed to retrieve access token: %s', content)
        error_msg = 'Invalid response {0}.'.format(resp.status)
        try:
            d = json.loads(content)
            if 'error' in d:
                error_msg = d['error']
                if 'error_description' in d:
                    error_msg += ': ' + d['error_description']
                self.invalid = True
                if self.store is not None:
                    self.store.locked_put(self)
        except (TypeError, ValueError):
            pass
        raise HttpAccessTokenRefreshError(error_msg, status=resp.status)