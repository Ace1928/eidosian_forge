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
class TestReadonlyStore(TestStore):
    scenarios = [(key, {'get_store': builder}) for key, builder in config.test_store_builder_registry.iteritems()]

    def test_building_delays_load(self):
        store = self.get_store(self)
        self.assertEqual(False, store.is_loaded())
        store._load_from_string(b'')
        self.assertEqual(True, store.is_loaded())

    def test_get_no_sections_for_empty(self):
        store = self.get_store(self)
        store._load_from_string(b'')
        self.assertEqual([], list(store.get_sections()))

    def test_get_default_section(self):
        store = self.get_store(self)
        store._load_from_string(b'foo=bar')
        sections = list(store.get_sections())
        self.assertLength(1, sections)
        self.assertSectionContent((None, {'foo': 'bar'}), sections[0])

    def test_get_named_section(self):
        store = self.get_store(self)
        store._load_from_string(b'[baz]\nfoo=bar')
        sections = list(store.get_sections())
        self.assertLength(1, sections)
        self.assertSectionContent(('baz', {'foo': 'bar'}), sections[0])

    def test_load_from_string_fails_for_non_empty_store(self):
        store = self.get_store(self)
        store._load_from_string(b'foo=bar')
        self.assertRaises(AssertionError, store._load_from_string, b'bar=baz')