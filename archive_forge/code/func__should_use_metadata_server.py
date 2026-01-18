import hashlib
import hmac
import json
import os
import posixpath
import re
from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
def _should_use_metadata_server(self):
    if not os.environ.get(environment_vars.AWS_REGION) and (not os.environ.get(environment_vars.AWS_DEFAULT_REGION)):
        return True
    if not os.environ.get(environment_vars.AWS_ACCESS_KEY_ID) or not os.environ.get(environment_vars.AWS_SECRET_ACCESS_KEY):
        return True
    return False