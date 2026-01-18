import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def _get_authorized_user_credentials(filename, info, scopes=None):
    from google.oauth2 import credentials
    try:
        credentials = credentials.Credentials.from_authorized_user_info(info, scopes=scopes)
    except ValueError as caught_exc:
        msg = 'Failed to load authorized user credentials from {}'.format(filename)
        new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
        six.raise_from(new_exc, caught_exc)
    return (credentials, None)