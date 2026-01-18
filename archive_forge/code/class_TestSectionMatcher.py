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
class TestSectionMatcher(TestStore):
    scenarios = [('location', {'matcher': config.LocationMatcher}), ('id', {'matcher': config.NameMatcher})]

    def setUp(self):
        super().setUp()
        self.get_store = config.test_store_builder_registry.get('configobj')

    def test_no_matches_for_empty_stores(self):
        store = self.get_store(self)
        store._load_from_string(b'')
        matcher = self.matcher(store, '/bar')
        self.assertEqual([], list(matcher.get_sections()))

    def test_build_doesnt_load_store(self):
        store = self.get_store(self)
        self.matcher(store, '/bar')
        self.assertFalse(store.is_loaded())