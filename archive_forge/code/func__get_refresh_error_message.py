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
def _get_refresh_error_message(content):
    """Constructs an error from the http response.

    Args:
        response: http response
        content: parsed response content

    Returns:
        error message to show
    """
    error_msg = 'Invalid response.'
    if 'error' in content:
        error_msg = content['error']
        if 'error_description' in content:
            error_msg += ': ' + content['error_description']
    return error_msg