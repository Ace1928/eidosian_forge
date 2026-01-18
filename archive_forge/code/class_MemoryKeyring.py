import datetime
from unittest import mock
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient import utils as client_utils
class MemoryKeyring(keyring.backend.KeyringBackend):
    """A Simple testing keyring.

            This class supports stubbing an initial password to be returned by
            setting password, and allows easy password and key retrieval. Also
            records if a password was retrieved.
            """

    def __init__(self):
        self.key = None
        self.password = None
        self.fetched = False
        self.get_password_called = False
        self.set_password_called = False

    def supported(self):
        return 1

    def get_password(self, service, username):
        self.get_password_called = True
        key = username + '@' + service
        if self.key and self.key != key:
            return None
        if self.password:
            self.fetched = True
        return self.password

    def set_password(self, service, username, password):
        self.set_password_called = True
        self.key = username + '@' + service
        self.password = password