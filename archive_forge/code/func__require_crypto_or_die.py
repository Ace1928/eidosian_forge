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
def _require_crypto_or_die():
    """Ensure we have a crypto library, or throw CryptoUnavailableError.

    The oauth2client.crypt module requires either PyCrypto or PyOpenSSL
    to be available in order to function, but these are optional
    dependencies.
    """
    if not HAS_CRYPTO:
        raise CryptoUnavailableError('No crypto library available')