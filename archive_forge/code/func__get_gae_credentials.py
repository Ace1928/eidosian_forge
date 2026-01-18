import io
import json
import os
import six
from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions
def _get_gae_credentials():
    """Gets Google App Engine App Identity credentials and project ID."""
    return _default._get_gae_credentials()