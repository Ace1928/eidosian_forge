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
class TestMemoryStack(tests.TestCase):

    def test_get(self):
        conf = config.MemoryStack(b'foo=bar')
        self.assertEqual('bar', conf.get('foo'))

    def test_set(self):
        conf = config.MemoryStack(b'foo=bar')
        conf.set('foo', 'baz')
        self.assertEqual('baz', conf.get('foo'))

    def test_no_content(self):
        conf = config.MemoryStack()
        self.assertFalse(conf.store.is_loaded())
        self.assertRaises(NotImplementedError, conf.get, 'foo')
        conf.store._load_from_string(b'foo=bar')
        self.assertEqual('bar', conf.get('foo'))