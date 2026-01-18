import io
import json
import os
import six
from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions
def _get_gce_credentials(request=None):
    """Gets credentials and project ID from the GCE Metadata Service."""
    return _default._get_gce_credentials(request)