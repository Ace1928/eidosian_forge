import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
@staticmethod
def _CreateDevAppServerCookieData(email, admin):
    """Creates cookie payload data.

    Args:
      email: The user's email address.
      admin: True if the user is an admin; False otherwise.

    Returns:
      String containing the cookie payload.
    """
    if email:
        user_id_digest = hashlib.md5(email.lower()).digest()
        user_id = '1' + ''.join(['%02d' % x for x in six_subset.iterbytes(user_id_digest)])[:20]
    else:
        user_id = ''
    return '%s:%s:%s' % (email, bool(admin), user_id)