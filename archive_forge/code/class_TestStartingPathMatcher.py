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
class TestStartingPathMatcher(TestStore):

    def setUp(self):
        super().setUp()
        self.store = config.IniFileStore()

    def assertSectionIDs(self, expected, location, content):
        self.store._load_from_string(content)
        matcher = config.StartingPathMatcher(self.store, location)
        sections = list(matcher.get_sections())
        self.assertLength(len(expected), sections)
        self.assertEqual(expected, [section.id for _, section in sections])
        return sections

    def test_empty(self):
        self.assertSectionIDs([], self.get_url(), b'')

    def test_url_vs_local_paths(self):
        self.assertSectionIDs(['/foo/bar', '/foo'], 'file:///foo/bar/baz', b'[/foo]\n[/foo/bar]\n')

    def test_local_path_vs_url(self):
        self.assertSectionIDs(['file:///foo/bar', 'file:///foo'], '/foo/bar/baz', b'[file:///foo]\n[file:///foo/bar]\n')

    def test_no_name_section_included_when_present(self):
        sections = self.assertSectionIDs(['/foo/bar', '/foo', None], '/foo/bar/baz', b'option = defined so the no-name section exists\n[/foo]\n[/foo/bar]\n')
        self.assertEqual(['baz', 'bar/baz', '/foo/bar/baz'], [s.locals['relpath'] for _, s in sections])

    def test_order_reversed(self):
        self.assertSectionIDs(['/foo/bar', '/foo'], '/foo/bar/baz', b'[/foo]\n[/foo/bar]\n')

    def test_unrelated_section_excluded(self):
        self.assertSectionIDs(['/foo/bar', '/foo'], '/foo/bar/baz', b'[/foo]\n[/foo/qux]\n[/foo/bar]\n')

    def test_glob_included(self):
        sections = self.assertSectionIDs(['/foo/*/baz', '/foo/b*', '/foo'], '/foo/bar/baz', b'[/foo]\n[/foo/qux]\n[/foo/b*]\n[/foo/*/baz]\n')
        self.assertEqual(['', 'baz', 'bar/baz'], [s.locals['relpath'] for _, s in sections])

    def test_respect_order(self):
        self.assertSectionIDs(['/foo', '/foo/b*', '/foo/*/baz'], '/foo/bar/baz', b'[/foo/*/baz]\n[/foo/qux]\n[/foo/b*]\n[/foo]\n')