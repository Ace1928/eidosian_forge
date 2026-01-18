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
class TestIniFileStoreContent(tests.TestCaseWithTransport):
    """Simulate loading a config store with content of various encodings.

    All files produced by bzr are in utf8 content.

    Users may modify them manually and end up with a file that can't be
    loaded. We need to issue proper error messages in this case.
    """
    invalid_utf8_char = b'\xff'

    def test_load_utf8(self):
        """Ensure we can load an utf8-encoded file."""
        t = self.get_transport()
        unicode_user = 'b€ar'
        unicode_content = 'user={}'.format(unicode_user)
        utf8_content = unicode_content.encode('utf8')
        t.put_bytes('foo.conf', utf8_content)
        store = config.TransportIniFileStore(t, 'foo.conf')
        store.load()
        stack = config.Stack([store.get_sections], store)
        self.assertEqual(unicode_user, stack.get('user'))

    def test_load_non_ascii(self):
        """Ensure we display a proper error on non-ascii, non utf-8 content."""
        t = self.get_transport()
        t.put_bytes('foo.conf', b'user=foo\n#%s\n' % (self.invalid_utf8_char,))
        store = config.TransportIniFileStore(t, 'foo.conf')
        self.assertRaises(config.ConfigContentError, store.load)

    def test_load_erroneous_content(self):
        """Ensure we display a proper error on content that can't be parsed."""
        t = self.get_transport()
        t.put_bytes('foo.conf', b'[open_section\n')
        store = config.TransportIniFileStore(t, 'foo.conf')
        self.assertRaises(config.ParseConfigError, store.load)

    def test_load_permission_denied(self):
        """Ensure we get warned when trying to load an inaccessible file."""
        warnings = []

        def warning(*args):
            warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)
        t = self.get_transport()

        def get_bytes(relpath):
            raise errors.PermissionDenied(relpath, '')
        t.get_bytes = get_bytes
        store = config.TransportIniFileStore(t, 'foo.conf')
        self.assertRaises(errors.PermissionDenied, store.load)
        self.assertEqual(warnings, ['Permission denied while trying to load configuration store %s.' % store.external_url()])