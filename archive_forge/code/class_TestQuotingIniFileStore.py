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
class TestQuotingIniFileStore(tests.TestCaseWithTransport):

    def get_store(self):
        return config.TransportIniFileStore(self.get_transport(), 'foo.conf')

    def test_get_quoted_string(self):
        store = self.get_store()
        store._load_from_string(b'foo= " abc "')
        stack = config.Stack([store.get_sections])
        self.assertEqual(' abc ', stack.get('foo'))

    def test_set_quoted_string(self):
        store = self.get_store()
        stack = config.Stack([store.get_sections], store)
        stack.set('foo', ' a b c ')
        store.save()
        self.assertFileEqual(b'foo = " a b c "' + os.linesep.encode('ascii'), 'foo.conf')