import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _extract_id_token(id_token):
    """Extract the JSON payload from a JWT.

    Does the extraction w/o checking the signature.

    Args:
        id_token: string or bytestring, OAuth 2.0 id_token.

    Returns:
        object, The deserialized JSON payload.
    """
    if type(id_token) == bytes:
        segments = id_token.split(b'.')
    else:
        segments = id_token.split(u'.')
    if len(segments) != 3:
        raise VerifyJwtTokenError('Wrong number of segments in token: {0}'.format(id_token))
    return json.loads(_helpers._from_bytes(_helpers._urlsafe_b64decode(segments[1])))