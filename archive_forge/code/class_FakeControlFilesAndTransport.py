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
class FakeControlFilesAndTransport:

    def __init__(self):
        self.files = {}
        self._transport = self

    def get(self, filename):
        try:
            return BytesIO(self.files[filename])
        except KeyError:
            raise _mod_transport.NoSuchFile(filename)

    def get_bytes(self, filename):
        try:
            return self.files[filename]
        except KeyError:
            raise _mod_transport.NoSuchFile(filename)

    def put(self, filename, fileobj):
        self.files[filename] = fileobj.read()

    def put_file(self, filename, fileobj):
        return self.put(filename, fileobj)