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
class TestStackExpandSectionLocals(tests.TestCaseWithTransport):

    def test_expand_locals_empty(self):
        l_store = config.LocationStore()
        l_store._load_from_string(b'\n[/home/user/project]\nbase = {basename}\nrel = {relpath}\n')
        l_store.save()
        stack = config.LocationStack('/home/user/project/')
        self.assertEqual('', stack.get('base', expand=True))
        self.assertEqual('', stack.get('rel', expand=True))

    def test_expand_basename_locally(self):
        l_store = config.LocationStore()
        l_store._load_from_string(b'\n[/home/user/project]\nbfoo = {basename}\n')
        l_store.save()
        stack = config.LocationStack('/home/user/project/branch')
        self.assertEqual('branch', stack.get('bfoo', expand=True))

    def test_expand_basename_locally_longer_path(self):
        l_store = config.LocationStore()
        l_store._load_from_string(b'\n[/home/user]\nbfoo = {basename}\n')
        l_store.save()
        stack = config.LocationStack('/home/user/project/dir/branch')
        self.assertEqual('branch', stack.get('bfoo', expand=True))

    def test_expand_relpath_locally(self):
        l_store = config.LocationStore()
        l_store._load_from_string(b'\n[/home/user/project]\nlfoo = loc-foo/{relpath}\n')
        l_store.save()
        stack = config.LocationStack('/home/user/project/branch')
        self.assertEqual('loc-foo/branch', stack.get('lfoo', expand=True))

    def test_expand_relpath_unknonw_in_global(self):
        g_store = config.GlobalStore()
        g_store._load_from_string(b'\n[DEFAULT]\ngfoo = {relpath}\n')
        g_store.save()
        stack = config.LocationStack('/home/user/project/branch')
        self.assertRaises(config.ExpandingUnknownOption, stack.get, 'gfoo', expand=True)

    def test_expand_local_option_locally(self):
        l_store = config.LocationStore()
        l_store._load_from_string(b'\n[/home/user/project]\nlfoo = loc-foo/{relpath}\nlbar = {gbar}\n')
        l_store.save()
        g_store = config.GlobalStore()
        g_store._load_from_string(b'\n[DEFAULT]\ngfoo = {lfoo}\ngbar = glob-bar\n')
        g_store.save()
        stack = config.LocationStack('/home/user/project/branch')
        self.assertEqual('glob-bar', stack.get('lbar', expand=True))
        self.assertEqual('loc-foo/branch', stack.get('gfoo', expand=True))

    def test_locals_dont_leak(self):
        """Make sure we chose the right local in presence of several sections.
        """
        l_store = config.LocationStore()
        l_store._load_from_string(b'\n[/home/user]\nlfoo = loc-foo/{relpath}\n[/home/user/project]\nlfoo = loc-foo/{relpath}\n')
        l_store.save()
        stack = config.LocationStack('/home/user/project/branch')
        self.assertEqual('loc-foo/branch', stack.get('lfoo', expand=True))
        stack = config.LocationStack('/home/user/bar/baz')
        self.assertEqual('loc-foo/bar/baz', stack.get('lfoo', expand=True))