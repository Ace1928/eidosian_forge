import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class UnencodedInMemoryKeyring(InMemoryKeyring):

    def get_password(self, service, username):
        pw = super(UnencodedInMemoryKeyring, self).get_password(service, username)
        return b64decode(pw[5:])