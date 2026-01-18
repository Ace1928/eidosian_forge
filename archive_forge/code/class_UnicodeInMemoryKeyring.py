import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class UnicodeInMemoryKeyring(InMemoryKeyring):

    def get_password(self, service, username):
        password = super(UnicodeInMemoryKeyring, self).get_password(service, username)
        if isinstance(password, unicode_type):
            password = password.encode('utf-8')
        return password