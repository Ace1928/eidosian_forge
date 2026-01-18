import datetime
import json
import os
from six.moves import http_client
from six.moves.urllib import parse as urlparse
from oauth2client import _helpers
from oauth2client import client
from oauth2client import transport
Fetch an oauth token for the

    Args:
        http: an object to be used to make HTTP requests.
        service_account: An email specifying the service account this token
            should represent. Default will be a token for the "default" service
            account of the current compute engine instance.

    Returns:
         A tuple of (access token, token expiration), where access token is the
         access token as a string and token expiration is a datetime object
         that indicates when the access token will expire.
    