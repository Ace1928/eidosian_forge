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
class TestStoreQuoting(TestStore):
    scenarios = [(key, {'get_store': builder}) for key, builder in config.test_store_builder_registry.iteritems()]

    def setUp(self):
        super().setUp()
        self.store = self.get_store(self)
        self.store._load_from_string(b'')

    def assertIdempotent(self, s):
        """Assert that quoting an unquoted string is a no-op and vice-versa.

        What matters here is that option values, as they appear in a store, can
        be safely round-tripped out of the store and back.

        :param s: A string, quoted if required.
        """
        self.assertEqual(s, self.store.quote(self.store.unquote(s)))
        self.assertEqual(s, self.store.unquote(self.store.quote(s)))

    def test_empty_string(self):
        if isinstance(self.store, config.IniFileStore):
            self.assertRaises(AssertionError, self.assertIdempotent, '')
        else:
            self.assertIdempotent('')
        self.assertIdempotent('""')

    def test_embedded_spaces(self):
        self.assertIdempotent('" a b c "')

    def test_embedded_commas(self):
        self.assertIdempotent('" a , b c "')

    def test_simple_comma(self):
        if isinstance(self.store, config.IniFileStore):
            self.assertRaises(AssertionError, self.assertIdempotent, ',')
        else:
            self.assertIdempotent(',')
        self.assertIdempotent('","')

    def test_list(self):
        if isinstance(self.store, config.IniFileStore):
            self.assertRaises(AssertionError, self.assertIdempotent, 'a,b')
        else:
            self.assertIdempotent('a,b')