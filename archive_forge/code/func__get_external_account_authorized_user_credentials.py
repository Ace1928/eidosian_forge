import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def _get_external_account_authorized_user_credentials(filename, info, scopes=None, default_scopes=None, request=None):
    try:
        from google.auth import external_account_authorized_user
        credentials = external_account_authorized_user.Credentials.from_info(info)
    except ValueError:
        raise exceptions.DefaultCredentialsError('Failed to load external account authorized user credentials from {}'.format(filename))
    return (credentials, None)