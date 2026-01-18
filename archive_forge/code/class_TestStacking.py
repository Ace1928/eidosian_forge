import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class TestStacking(tests.TestCaseWithTransport):
    """Tests for operations on stacked remote repositories.

    The underlying format type must support stacking.
    """

    def test_access_stacked_remote(self):
        base_builder = self.make_branch_builder('base', format='1.9')
        base_builder.start_series()
        base_revid = base_builder.build_snapshot(None, [('add', ('', None, 'directory', None))], 'message', revision_id=b'rev-id')
        base_builder.finish_series()
        stacked_branch = self.make_branch('stacked', format='1.9')
        stacked_branch.set_stacked_on_url('../base')
        smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(smart_server)
        remote_bzrdir = BzrDir.open(smart_server.get_url() + '/stacked')
        remote_branch = remote_bzrdir.open_branch()
        remote_repo = remote_branch.repository
        remote_repo.lock_read()
        try:
            self.assertLength(1, remote_repo._fallback_repositories)
            self.assertIsInstance(remote_repo._fallback_repositories[0], RemoteRepository)
            self.assertTrue(remote_repo.has_revisions([base_revid]))
            self.assertTrue(remote_repo.has_revision(base_revid))
            self.assertEqual(remote_repo.get_revision(base_revid).message, 'message')
        finally:
            remote_repo.unlock()

    def prepare_stacked_remote_branch(self):
        """Get stacked_upon and stacked branches with content in each."""
        self.setup_smart_server_with_call_log()
        tree1 = self.make_branch_and_tree('tree1', format='1.9')
        tree1.commit('rev1', rev_id=b'rev1')
        tree2 = tree1.branch.controldir.sprout('tree2', stacked=True).open_workingtree()
        local_tree = tree2.branch.create_checkout('local')
        local_tree.commit('local changes make me feel good.')
        branch2 = Branch.open(self.get_url('tree2'))
        branch2.lock_read()
        self.addCleanup(branch2.unlock)
        return (tree1.branch, branch2)

    def test_stacked_get_parent_map(self):
        _, branch = self.prepare_stacked_remote_branch()
        repo = branch.repository
        self.assertEqual({b'rev1'}, set(repo.get_parent_map([b'rev1'])))

    def test_unstacked_get_parent_map(self):
        _, branch = self.prepare_stacked_remote_branch()
        provider = branch.repository._unstacked_provider
        self.assertEqual(set(), set(provider.get_parent_map([b'rev1'])))

    def fetch_stream_to_rev_order(self, stream):
        result = []
        for kind, substream in stream:
            if not kind == 'revisions':
                list(substream)
            else:
                for content in substream:
                    result.append(content.key[-1])
        return result

    def get_ordered_revs(self, format, order, branch_factory=None):
        """Get a list of the revisions in a stream to format format.

        :param format: The format of the target.
        :param order: the order that target should have requested.
        :param branch_factory: A callable to create a trunk and stacked branch
            to fetch from. If none, self.prepare_stacked_remote_branch is used.
        :result: The revision ids in the stream, in the order seen,
            the topological order of revisions in the source.
        """
        unordered_format = controldir.format_registry.get(format)()
        target_repository_format = unordered_format.repository_format
        self.assertEqual(order, target_repository_format._fetch_order)
        if branch_factory is None:
            branch_factory = self.prepare_stacked_remote_branch
        _, stacked = branch_factory()
        source = stacked.repository._get_source(target_repository_format)
        tip = stacked.last_revision()
        stacked.repository._ensure_real()
        graph = stacked.repository.get_graph()
        revs = [r for r, ps in graph.iter_ancestry([tip]) if r != NULL_REVISION]
        revs.reverse()
        search = vf_search.PendingAncestryResult([tip], stacked.repository)
        self.reset_smart_call_log()
        stream = source.get_stream(search)
        return (self.fetch_stream_to_rev_order(stream), revs)

    def test_stacked_get_stream_unordered(self):
        rev_ord, expected_revs = self.get_ordered_revs('1.9', 'unordered')
        self.assertEqual(set(expected_revs), set(rev_ord))
        self.assertLength(2, self.hpss_calls)

    def test_stacked_on_stacked_get_stream_unordered(self):

        def make_stacked_stacked():
            _, stacked = self.prepare_stacked_remote_branch()
            tree = stacked.controldir.sprout('tree3', stacked=True).open_workingtree()
            local_tree = tree.branch.create_checkout('local-tree3')
            local_tree.commit('more local changes are better')
            branch = Branch.open(self.get_url('tree3'))
            branch.lock_read()
            self.addCleanup(branch.unlock)
            return (None, branch)
        rev_ord, expected_revs = self.get_ordered_revs('1.9', 'unordered', branch_factory=make_stacked_stacked)
        self.assertEqual(set(expected_revs), set(rev_ord))
        self.assertLength(3, self.hpss_calls)

    def test_stacked_get_stream_topological(self):
        rev_ord, expected_revs = self.get_ordered_revs('knit', 'topological')
        self.assertEqual(expected_revs, rev_ord)
        self.assertLength(14, self.hpss_calls)

    def test_stacked_get_stream_groupcompress(self):
        raise tests.TestSkipped('No groupcompress ordered format available')
        rev_ord, expected_revs = self.get_ordered_revs('dev5', 'groupcompress')
        self.assertEqual(expected_revs, reversed(rev_ord))
        self.assertLength(2, self.hpss_calls)

    def test_stacked_pull_more_than_stacking_has_bug_360791(self):
        self.setup_smart_server_with_call_log()
        trunk = self.make_branch_and_tree('trunk', format='1.9-rich-root')
        trunk.commit('start')
        stacked_branch = trunk.branch.create_clone_on_transport(self.get_transport('stacked'), stacked_on=trunk.branch.base)
        local = self.make_branch('local', format='1.9-rich-root')
        local.repository.fetch(stacked_branch.repository, stacked_branch.last_revision())