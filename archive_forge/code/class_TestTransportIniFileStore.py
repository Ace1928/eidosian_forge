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
class TestTransportIniFileStore(TestStore):

    def test_loading_unknown_file_fails(self):
        store = config.TransportIniFileStore(self.get_transport(), 'I-do-not-exist')
        self.assertRaises(_mod_transport.NoSuchFile, store.load)

    def test_invalid_content(self):
        store = config.TransportIniFileStore(self.get_transport(), 'foo.conf')
        self.assertEqual(False, store.is_loaded())
        exc = self.assertRaises(config.ParseConfigError, store._load_from_string, b'this is invalid !')
        self.assertEndsWith(exc.filename, 'foo.conf')
        self.assertEqual(False, store.is_loaded())

    def test_get_embedded_sections(self):
        store = config.TransportIniFileStore(self.get_transport(), 'foo.conf')
        store._load_from_string(b'\nfoo=bar\nl=1,2\n[DEFAULT]\nfoo_in_DEFAULT=foo_DEFAULT\n[bar]\nfoo_in_bar=barbar\n[baz]\nfoo_in_baz=barbaz\n[[qux]]\nfoo_in_qux=quux\n')
        sections = list(store.get_sections())
        self.assertLength(4, sections)
        self.assertSectionContent((None, {'foo': 'bar', 'l': '1,2'}), sections[0])
        self.assertSectionContent(('DEFAULT', {'foo_in_DEFAULT': 'foo_DEFAULT'}), sections[1])
        self.assertSectionContent(('bar', {'foo_in_bar': 'barbar'}), sections[2])
        self.assertSectionContent(('baz', {'foo_in_baz': 'barbaz', 'qux': {'foo_in_qux': 'quux'}}), sections[3])