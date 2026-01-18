from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from google_reauth import challenges
from google_reauth import errors
from google_reauth import _helpers
from google_reauth import _reauth_client
from six.moves import http_client
from six.moves import range
def _substr_for_error_message(content):
    """Returns content string to include in the error message"""
    return content if len(content) <= 100 else content[0:97] + '...'