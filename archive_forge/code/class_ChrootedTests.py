import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
class ChrootedTests(TestCaseWithTransport):
    """A support class that provides readonly urls outside the local namespace.

    This is done by checking if self.transport_server is a MemoryServer. if it
    is then we are chrooted already, if it is not then an HttpServer is used
    for readonly urls.
    """

    def setUp(self):
        super().setUp()
        if not self.vfs_transport_factory == memory.MemoryServer:
            self.transport_readonly_server = http_server.HttpServer

    def local_branch_path(self, branch):
        return os.path.realpath(urlutils.local_path_from_url(branch.base))

    def test_open_containing(self):
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing, self.get_readonly_url(''))
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing, self.get_readonly_url('g/p/q'))
        control = bzrdir.BzrDir.create(self.get_url())
        branch, relpath = bzrdir.BzrDir.open_containing(self.get_readonly_url(''))
        self.assertEqual('', relpath)
        branch, relpath = bzrdir.BzrDir.open_containing(self.get_readonly_url('g/p/q'))
        self.assertEqual('g/p/q', relpath)

    def test_open_containing_tree_branch_or_repository_empty(self):
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open_containing_tree_branch_or_repository, self.get_readonly_url(''))

    def test_open_containing_tree_branch_or_repository_all(self):
        self.make_branch_and_tree('topdir')
        tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('topdir/foo')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('topdir', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'foo')

    def test_open_containing_tree_branch_or_repository_no_tree(self):
        self.make_branch('branch')
        tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('branch/foo')
        self.assertEqual(tree, None)
        self.assertEqual(os.path.realpath('branch'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('branch', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'foo')

    def test_open_containing_tree_branch_or_repository_repo(self):
        self.make_repository('repo')
        tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('repo')
        self.assertEqual(tree, None)
        self.assertEqual(branch, None)
        self.assertEqual(osutils.realpath(os.path.join('repo', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
        self.assertEqual(relpath, '')

    def test_open_containing_tree_branch_or_repository_shared_repo(self):
        self.make_repository('shared', shared=True)
        bzrdir.BzrDir.create_branch_convenience('shared/branch', force_new_tree=False)
        tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('shared/branch')
        self.assertEqual(tree, None)
        self.assertEqual(os.path.realpath('shared/branch'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('shared', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
        self.assertEqual(relpath, '')

    def test_open_containing_tree_branch_or_repository_branch_subdir(self):
        self.make_branch_and_tree('foo')
        self.build_tree(['foo/bar/'])
        tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('foo/bar')
        self.assertEqual(os.path.realpath('foo'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('foo'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('foo', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'bar')

    def test_open_containing_tree_branch_or_repository_repo_subdir(self):
        self.make_repository('bar')
        self.build_tree(['bar/baz/'])
        tree, branch, repo, relpath = bzrdir.BzrDir.open_containing_tree_branch_or_repository('bar/baz')
        self.assertEqual(tree, None)
        self.assertEqual(branch, None)
        self.assertEqual(osutils.realpath(os.path.join('bar', '.bzr', 'repository')), repo.controldir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'baz')

    def test_open_containing_from_transport(self):
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing_from_transport, _mod_transport.get_transport_from_url(self.get_readonly_url('')))
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing_from_transport, _mod_transport.get_transport_from_url(self.get_readonly_url('g/p/q')))
        control = bzrdir.BzrDir.create(self.get_url())
        branch, relpath = bzrdir.BzrDir.open_containing_from_transport(_mod_transport.get_transport_from_url(self.get_readonly_url('')))
        self.assertEqual('', relpath)
        branch, relpath = bzrdir.BzrDir.open_containing_from_transport(_mod_transport.get_transport_from_url(self.get_readonly_url('g/p/q')))
        self.assertEqual('g/p/q', relpath)

    def test_open_containing_tree_or_branch(self):
        self.make_branch_and_tree('topdir')
        tree, branch, relpath = bzrdir.BzrDir.open_containing_tree_or_branch('topdir/foo')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertIs(tree.controldir, branch.controldir)
        self.assertEqual('foo', relpath)
        tree, branch, relpath = bzrdir.BzrDir.open_containing_tree_or_branch(self.get_readonly_url('topdir/foo'))
        self.assertEqual(None, tree)
        self.assertEqual('foo', relpath)
        self.make_branch('topdir/foo')
        tree, branch, relpath = bzrdir.BzrDir.open_containing_tree_or_branch('topdir/foo')
        self.assertIs(tree, None)
        self.assertEqual(os.path.realpath('topdir/foo'), self.local_branch_path(branch))
        self.assertEqual('', relpath)

    def test_open_tree_or_branch(self):
        self.make_branch_and_tree('topdir')
        tree, branch = bzrdir.BzrDir.open_tree_or_branch('topdir')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertIs(tree.controldir, branch.controldir)
        tree, branch = bzrdir.BzrDir.open_tree_or_branch(self.get_readonly_url('topdir'))
        self.assertEqual(None, tree)
        self.make_branch('topdir/foo')
        tree, branch = bzrdir.BzrDir.open_tree_or_branch('topdir/foo')
        self.assertIs(tree, None)
        self.assertEqual(os.path.realpath('topdir/foo'), self.local_branch_path(branch))

    def test_open_tree_or_branch_named(self):
        tree = self.make_branch_and_tree('topdir')
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_tree_or_branch, 'topdir', name='missing')
        tree.branch.controldir.create_branch('named')
        tree, branch = bzrdir.BzrDir.open_tree_or_branch('topdir', name='named')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertEqual(branch.name, 'named')
        self.assertIs(tree.controldir, branch.controldir)

    def test_open_from_transport(self):
        control = bzrdir.BzrDir.create(self.get_url())
        t = self.get_transport()
        opened_bzrdir = bzrdir.BzrDir.open_from_transport(t)
        self.assertEqual(t.base, opened_bzrdir.root_transport.base)
        self.assertIsInstance(opened_bzrdir, bzrdir.BzrDir)

    def test_open_from_transport_no_bzrdir(self):
        t = self.get_transport()
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_from_transport, t)

    def test_open_from_transport_bzrdir_in_parent(self):
        control = bzrdir.BzrDir.create(self.get_url())
        t = self.get_transport()
        t.mkdir('subdir')
        t = t.clone('subdir')
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_from_transport, t)

    def test_sprout_recursive(self):
        tree = self.make_branch_and_tree('tree1')
        sub_tree = self.make_branch_and_tree('tree1/subtree')
        sub_tree.set_root_id(b'subtree-root')
        tree.add_reference(sub_tree)
        tree.set_reference_info('subtree', sub_tree.branch.user_url)
        self.build_tree(['tree1/subtree/file'])
        sub_tree.add('file')
        tree.commit('Initial commit')
        tree2 = tree.controldir.sprout('tree2').open_workingtree()
        tree2.lock_read()
        self.addCleanup(tree2.unlock)
        self.assertPathExists('tree2/subtree/file')
        self.assertEqual('tree-reference', tree2.kind('subtree'))

    def test_cloning_metadir(self):
        """Ensure that cloning metadir is suitable"""
        bzrdir = self.make_controldir('bzrdir')
        bzrdir.cloning_metadir()
        branch = self.make_branch('branch', format='knit')
        format = branch.controldir.cloning_metadir()
        self.assertIsInstance(format.workingtree_format, workingtree_4.WorkingTreeFormat6)

    def test_sprout_recursive_treeless(self):
        tree = self.make_branch_and_tree('tree1', format='development-subtree')
        sub_tree = self.make_branch_and_tree('tree1/subtree', format='development-subtree')
        tree.add_reference(sub_tree)
        tree.set_reference_info('subtree', sub_tree.branch.user_url)
        self.build_tree(['tree1/subtree/file'])
        sub_tree.add('file')
        tree.commit('Initial commit')
        tree.branch.get_config_stack().set('transform.orphan_policy', 'move')
        tree.controldir.destroy_workingtree()
        repo = self.make_repository('repo', shared=True, format='development-subtree')
        repo.set_make_working_trees(False)
        tree.controldir.sprout('repo/tree2')
        self.assertPathExists('repo/tree2/subtree')
        self.assertPathDoesNotExist('repo/tree2/subtree/file')

    def make_foo_bar_baz(self):
        foo = bzrdir.BzrDir.create_branch_convenience('foo').controldir
        bar = self.make_branch('foo/bar').controldir
        baz = self.make_branch('baz').controldir
        return (foo, bar, baz)

    def test_find_controldirs(self):
        foo, bar, baz = self.make_foo_bar_baz()
        t = self.get_transport()
        self.assertEqualBzrdirs([baz, foo, bar], bzrdir.BzrDir.find_controldirs(t))

    def make_fake_permission_denied_transport(self, transport, paths):
        """Create a transport that raises PermissionDenied for some paths."""

        def filter(path):
            if path in paths:
                raise errors.PermissionDenied(path)
            return path
        path_filter_server = pathfilter.PathFilteringServer(transport, filter)
        path_filter_server.start_server()
        self.addCleanup(path_filter_server.stop_server)
        path_filter_transport = pathfilter.PathFilteringTransport(path_filter_server, '.')
        return (path_filter_server, path_filter_transport)

    def assertBranchUrlsEndWith(self, expect_url_suffix, actual_bzrdirs):
        """Check that each branch url ends with the given suffix."""
        for actual_bzrdir in actual_bzrdirs:
            self.assertEndsWith(actual_bzrdir.user_url, expect_url_suffix)

    def test_find_controldirs_permission_denied(self):
        foo, bar, baz = self.make_foo_bar_baz()
        t = self.get_transport()
        path_filter_server, path_filter_transport = self.make_fake_permission_denied_transport(t, ['foo'])
        self.assertBranchUrlsEndWith('/baz/', bzrdir.BzrDir.find_controldirs(path_filter_transport))
        smart_transport = self.make_smart_server('.', backing_server=path_filter_server)
        self.assertBranchUrlsEndWith('/baz/', bzrdir.BzrDir.find_controldirs(smart_transport))

    def test_find_controldirs_list_current(self):

        def list_current(transport):
            return [s for s in transport.list_dir('') if s != 'baz']
        foo, bar, baz = self.make_foo_bar_baz()
        t = self.get_transport()
        self.assertEqualBzrdirs([foo, bar], bzrdir.BzrDir.find_controldirs(t, list_current=list_current))

    def test_find_controldirs_evaluate(self):

        def evaluate(bzrdir):
            try:
                repo = bzrdir.open_repository()
            except errors.NoRepositoryPresent:
                return (True, bzrdir.root_transport.base)
            else:
                return (False, bzrdir.root_transport.base)
        foo, bar, baz = self.make_foo_bar_baz()
        t = self.get_transport()
        self.assertEqual([baz.root_transport.base, foo.root_transport.base], list(bzrdir.BzrDir.find_controldirs(t, evaluate=evaluate)))

    def assertEqualBzrdirs(self, first, second):
        first = list(first)
        second = list(second)
        self.assertEqual(len(first), len(second))
        for x, y in zip(first, second):
            self.assertEqual(x.root_transport.base, y.root_transport.base)

    def test_find_branches(self):
        root = self.make_repository('', shared=True)
        foo, bar, baz = self.make_foo_bar_baz()
        qux = self.make_controldir('foo/qux')
        t = self.get_transport()
        branches = bzrdir.BzrDir.find_branches(t)
        self.assertEqual(baz.root_transport.base, branches[0].base)
        self.assertEqual(foo.root_transport.base, branches[1].base)
        self.assertEqual(bar.root_transport.base, branches[2].base)
        branches = bzrdir.BzrDir.find_branches(t.clone('foo'))
        self.assertEqual(foo.root_transport.base, branches[0].base)
        self.assertEqual(bar.root_transport.base, branches[1].base)