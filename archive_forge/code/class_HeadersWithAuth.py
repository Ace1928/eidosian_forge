import datetime
import json
import httplib2
from google.auth import aws
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from gslib.tests import testcase
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from six import add_move, MovedModule
from six.moves import mock
class HeadersWithAuth(dict):
    """A utility class to use to make sure a set of headers includes specific authentication"""

    def __init__(self, token):
        self.token = token or ''

    def __eq__(self, headers):
        return headers[b'Authorization'] == bytes('Bearer ' + self.token, 'utf-8')