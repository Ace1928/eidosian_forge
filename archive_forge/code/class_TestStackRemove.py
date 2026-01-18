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
class TestStackRemove(TestStackWithTransport):

    def test_remove_existing(self):
        conf = self.get_stack(self)
        conf.set('foo', 'bar')
        self.assertEqual('bar', conf.get('foo'))
        conf.remove('foo')
        self.assertEqual(None, conf.get('foo'))

    def test_remove_unknown(self):
        conf = self.get_stack(self)
        self.assertRaises(KeyError, conf.remove, 'I_do_not_exist')

    def test_remove_hook(self):
        calls = []

        def hook(*args):
            calls.append(args)
        config.ConfigHooks.install_named_hook('remove', hook, None)
        self.assertLength(0, calls)
        conf = self.get_stack(self)
        conf.set('foo', 'bar')
        conf.remove('foo')
        self.assertLength(1, calls)
        self.assertEqual((conf, 'foo'), calls[0])