import os
import platform
import socket
import stat
import six
from oauthlib import oauth1
from six.moves.urllib.parse import parse_qs, urlencode
from lazr.restfulclient.authorize import HttpAuthorizer
from lazr.restfulclient.errors import CredentialsFileError
class TruthyString(six.text_type):
    """A Unicode string which is always true."""

    def __bool__(self):
        return True
    __nonzero__ = __bool__