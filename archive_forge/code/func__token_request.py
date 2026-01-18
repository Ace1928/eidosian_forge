import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
def _token_request(self, request_body):
    """
        Return an updated token from a token request body.

        :param  request_body: A dictionary of values to send in the body of the
                              token request.
        :type   request_body: ``dict``

        :return:  A dictionary with updated token information
        :rtype:   ``dict``
        """
    data = urlencode(request_body)
    try:
        response = self.request('/o/oauth2/token', method='POST', data=data)
    except AttributeError:
        raise GoogleAuthError('Invalid authorization response, please check your credentials and time drift.')
    token_info = response.object
    if 'expires_in' in token_info:
        expire_time = _utcnow() + datetime.timedelta(seconds=token_info['expires_in'])
        token_info['expire_time'] = _utc_timestamp(expire_time)
    return token_info