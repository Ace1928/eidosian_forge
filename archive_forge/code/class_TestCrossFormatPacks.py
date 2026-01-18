from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class TestCrossFormatPacks(TestCaseWithTransport):

    def log_pack(self, hint=None):
        self.calls.append(('pack', hint))
        self.orig_pack(hint=hint)
        if self.expect_hint:
            self.assertTrue(hint)

    def run_stream(self, src_fmt, target_fmt, expect_pack_called):
        self.expect_hint = expect_pack_called
        self.calls = []
        source_tree = self.make_branch_and_tree('src', format=src_fmt)
        source_tree.lock_write()
        self.addCleanup(source_tree.unlock)
        tip = source_tree.commit('foo')
        target = self.make_repository('target', format=target_fmt)
        target.lock_write()
        self.addCleanup(target.unlock)
        source = source_tree.branch.repository._get_source(target._format)
        self.orig_pack = target.pack
        self.overrideAttr(target, 'pack', self.log_pack)
        search = target.search_missing_revision_ids(source_tree.branch.repository, revision_ids=[tip])
        stream = source.get_stream(search)
        from_format = source_tree.branch.repository._format
        sink = target._get_sink()
        sink.insert_stream(stream, from_format, [])
        if expect_pack_called:
            self.assertLength(1, self.calls)
        else:
            self.assertLength(0, self.calls)

    def run_fetch(self, src_fmt, target_fmt, expect_pack_called):
        self.expect_hint = expect_pack_called
        self.calls = []
        source_tree = self.make_branch_and_tree('src', format=src_fmt)
        source_tree.lock_write()
        self.addCleanup(source_tree.unlock)
        source_tree.commit('foo')
        target = self.make_repository('target', format=target_fmt)
        target.lock_write()
        self.addCleanup(target.unlock)
        source = source_tree.branch.repository
        self.orig_pack = target.pack
        self.overrideAttr(target, 'pack', self.log_pack)
        target.fetch(source)
        if expect_pack_called:
            self.assertLength(1, self.calls)
        else:
            self.assertLength(0, self.calls)

    def test_sink_format_hint_no(self):
        self.run_stream('1.9', 'rich-root-pack', False)

    def test_sink_format_hint_yes(self):
        self.run_stream('1.9', '2a', True)

    def test_sink_format_same_no(self):
        self.run_stream('2a', '2a', False)

    def test_IDS_format_hint_no(self):
        self.run_fetch('1.9', 'rich-root-pack', False)

    def test_IDS_format_hint_yes(self):
        self.run_fetch('1.9', '2a', True)

    def test_IDS_format_same_no(self):
        self.run_fetch('2a', '2a', False)