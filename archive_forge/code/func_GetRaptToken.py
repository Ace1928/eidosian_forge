import base64
import getpass
import json
import logging
import sys
from oauth2client.contrib import reauth_errors
from pyu2f import errors as u2ferrors
from pyu2f import model
from pyu2f.convenience import authenticator
from six.moves import urllib
def GetRaptToken(http_request, client_id, client_secret, refresh_token, token_uri, scopes=None):
    """Given an http request method and refresh_token, get rapt token."""
    GetPrintCallback()('Reauthentication required.\n')
    query_params = {'client_id': client_id, 'client_secret': client_secret, 'refresh_token': refresh_token, 'scope': REAUTH_SCOPE, 'grant_type': 'refresh_token'}
    _, content = http_request(token_uri, method='POST', body=urllib.parse.urlencode(query_params), headers={'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        reauth_access_token = json.loads(content)['access_token']
    except (ValueError, KeyError):
        raise reauth_errors.ReauthAccessTokenRefreshError
    rapt_token = ObtainRapt(http_request, reauth_access_token, requested_scopes=scopes)
    return rapt_token