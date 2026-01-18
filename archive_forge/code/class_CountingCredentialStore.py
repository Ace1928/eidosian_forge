import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class CountingCredentialStore(config.CredentialStore):

    def __init__(self):
        self._calls = 0

    def get_credentials(self, scheme, host, port=None, user=None, path=None, realm=None):
        self._calls += 1
        return None