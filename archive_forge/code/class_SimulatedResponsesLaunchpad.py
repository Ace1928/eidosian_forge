from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
class SimulatedResponsesLaunchpad(Launchpad):
    responses = []

    def httpFactory(self, *args):
        return SimulatedResponsesHttp(deque(self.responses), self, self.authorization_engine, *args)

    @classmethod
    def credential_store_factory(cls, credential_save_failed):
        return UnencryptedFileCredentialStore(tempfile.mkstemp()[1], credential_save_failed)