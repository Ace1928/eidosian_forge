from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.core.util import files
import requests
@staticmethod
def _chkpath(method, path):
    """Return an HTTP status for the given filesystem path."""
    if method.lower() not in ('get', 'head'):
        return (requests.codes.not_allowed, 'Method Not Allowed')
    elif os.path.isdir(path):
        return (requests.codes.bad_request, 'Path Not A File')
    elif not os.path.isfile(path):
        return (requests.codes.not_found, 'File Not Found')
    elif not os.access(path, os.R_OK):
        return (requests.codes.forbidden, 'Access Denied')
    else:
        return (requests.codes.ok, 'OK')