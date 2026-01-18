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
class TestStackIterSections(tests.TestCase):

    def test_empty_stack(self):
        conf = config.Stack([])
        sections = list(conf.iter_sections())
        self.assertLength(0, sections)

    def test_empty_store(self):
        store = config.IniFileStore()
        store._load_from_string(b'')
        conf = config.Stack([store.get_sections])
        sections = list(conf.iter_sections())
        self.assertLength(0, sections)

    def test_simple_store(self):
        store = config.IniFileStore()
        store._load_from_string(b'foo=bar')
        conf = config.Stack([store.get_sections])
        tuples = list(conf.iter_sections())
        self.assertLength(1, tuples)
        found_store, found_section = tuples[0]
        self.assertIs(store, found_store)

    def test_two_stores(self):
        store1 = config.IniFileStore()
        store1._load_from_string(b'foo=bar')
        store2 = config.IniFileStore()
        store2._load_from_string(b'bar=qux')
        conf = config.Stack([store1.get_sections, store2.get_sections])
        tuples = list(conf.iter_sections())
        self.assertLength(2, tuples)
        self.assertIs(store1, tuples[0][0])
        self.assertIs(store2, tuples[1][0])