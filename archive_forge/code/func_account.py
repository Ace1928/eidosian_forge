from datetime import datetime
import io
import json
import logging
import warnings
from google.auth import _cloud_sdk
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import metrics
from google.oauth2 import reauth
@property
def account(self):
    """str: The user account associated with the credential. If the account is unknown an empty string is returned."""
    return self._account