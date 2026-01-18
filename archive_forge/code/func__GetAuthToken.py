import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetAuthToken(self, email, password):
    """Uses ClientLogin to authenticate the user, returning an auth token.

    Args:
      email:    The user's email address
      password: The user's password

    Raises:
      ClientLoginError: If there was an error authenticating with ClientLogin.
      HTTPError: If there was some other form of HTTP error.

    Returns:
      The authentication token returned by ClientLogin.
    """
    account_type = self.account_type
    if not account_type:
        if self.host.split(':')[0].endswith('.google.com') or (self.host_override and self.host_override.split(':')[0].endswith('.google.com')):
            account_type = 'HOSTED_OR_GOOGLE'
        else:
            account_type = 'GOOGLE'
    data = {'Email': email, 'Passwd': password, 'service': 'ah', 'source': self.source, 'accountType': account_type}
    req = self._CreateRequest(url='https://%s/accounts/ClientLogin' % encoding.GetEncodedValue(os.environ, 'APPENGINE_AUTH_SERVER', 'www.google.com'), data=urlencode_fn(data))
    try:
        response = self.opener.open(req)
        response_body = response.read()
        response_dict = dict((x.split('=') for x in response_body.split('\n') if x))
        if encoding.GetEncodedValue(os.environ, 'APPENGINE_RPC_USE_SID', '0') == '1':
            self.extra_headers['Cookie'] = 'SID=%s; Path=/;' % response_dict['SID']
        return response_dict['Auth']
    except HTTPError as e:
        if e.code == 403:
            body = e.read()
            response_dict = dict((x.split('=', 1) for x in body.split('\n') if x))
            raise ClientLoginError(req.get_full_url(), e.code, e.msg, e.headers, response_dict)
        else:
            raise