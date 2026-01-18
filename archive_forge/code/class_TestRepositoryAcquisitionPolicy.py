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
class TestRepositoryAcquisitionPolicy(TestCaseWithTransport):

    def test_acquire_repository_standalone(self):
        """The default acquisition policy should create a standalone branch."""
        my_bzrdir = self.make_controldir('.')
        repo_policy = my_bzrdir.determine_repository_policy()
        repo, is_new = repo_policy.acquire_repository()
        self.assertEqual(repo.controldir.root_transport.base, my_bzrdir.root_transport.base)
        self.assertFalse(repo.is_shared())

    def test_determine_stacking_policy(self):
        parent_bzrdir = self.make_controldir('.')
        child_bzrdir = self.make_controldir('child')
        parent_bzrdir.get_config().set_default_stack_on('http://example.org')
        repo_policy = child_bzrdir.determine_repository_policy()
        self.assertEqual('http://example.org', repo_policy._stack_on)

    def test_determine_stacking_policy_relative(self):
        parent_bzrdir = self.make_controldir('.')
        child_bzrdir = self.make_controldir('child')
        parent_bzrdir.get_config().set_default_stack_on('child2')
        repo_policy = child_bzrdir.determine_repository_policy()
        self.assertEqual('child2', repo_policy._stack_on)
        self.assertEqual(parent_bzrdir.root_transport.base, repo_policy._stack_on_pwd)

    def prepare_default_stacking(self, child_format='1.6'):
        parent_bzrdir = self.make_controldir('.')
        child_branch = self.make_branch('child', format=child_format)
        parent_bzrdir.get_config().set_default_stack_on(child_branch.base)
        new_child_transport = parent_bzrdir.transport.clone('child2')
        return (child_branch, new_child_transport)

    def test_clone_on_transport_obeys_stacking_policy(self):
        child_branch, new_child_transport = self.prepare_default_stacking()
        new_child = child_branch.controldir.clone_on_transport(new_child_transport)
        self.assertEqual(child_branch.base, new_child.open_branch().get_stacked_on_url())

    def test_default_stacking_with_stackable_branch_unstackable_repo(self):
        source_bzrdir = self.make_controldir('source')
        knitpack_repo.RepositoryFormatKnitPack1().initialize(source_bzrdir)
        source_branch = breezy.bzr.branch.BzrBranchFormat7().initialize(source_bzrdir)
        parent_bzrdir = self.make_controldir('parent')
        stacked_on = self.make_branch('parent/stacked-on', format='pack-0.92')
        parent_bzrdir.get_config().set_default_stack_on(stacked_on.base)
        target = source_bzrdir.clone(self.get_url('parent/target'))

    def test_format_initialize_on_transport_ex_stacked_on(self):
        trunk = self.make_branch('trunk', format='1.9')
        t = self.get_transport('stacked')
        old_fmt = controldir.format_registry.make_controldir('pack-0.92')
        repo_name = old_fmt.repository_format.network_name()
        repo, control, require_stacking, repo_policy = old_fmt.initialize_on_transport_ex(t, repo_format_name=repo_name, stacked_on='../trunk', stack_on_pwd=t.base)
        if repo is not None:
            self.assertTrue(repo.is_write_locked())
            self.addCleanup(repo.unlock)
        else:
            repo = control.open_repository()
        self.assertIsInstance(control, bzrdir.BzrDir)
        opened = bzrdir.BzrDir.open(t.base)
        if not isinstance(old_fmt, remote.RemoteBzrDirFormat):
            self.assertEqual(control._format.network_name(), old_fmt.network_name())
            self.assertEqual(control._format.network_name(), opened._format.network_name())
        self.assertEqual(control.__class__, opened.__class__)
        self.assertLength(1, repo._fallback_repositories)

    def test_sprout_obeys_stacking_policy(self):
        child_branch, new_child_transport = self.prepare_default_stacking()
        new_child = child_branch.controldir.sprout(new_child_transport.base)
        self.assertEqual(child_branch.base, new_child.open_branch().get_stacked_on_url())

    def test_clone_ignores_policy_for_unsupported_formats(self):
        child_branch, new_child_transport = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.controldir.clone_on_transport(new_child_transport)
        self.assertRaises(branch.UnstackableBranchFormat, new_child.open_branch().get_stacked_on_url)

    def test_sprout_ignores_policy_for_unsupported_formats(self):
        child_branch, new_child_transport = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.controldir.sprout(new_child_transport.base)
        self.assertRaises(branch.UnstackableBranchFormat, new_child.open_branch().get_stacked_on_url)

    def test_sprout_upgrades_format_if_stacked_specified(self):
        child_branch, new_child_transport = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.controldir.sprout(new_child_transport.base, stacked=True)
        self.assertEqual(child_branch.controldir.root_transport.base, new_child.open_branch().get_stacked_on_url())
        repo = new_child.open_repository()
        self.assertTrue(repo._format.supports_external_lookups)
        self.assertFalse(repo.supports_rich_root())

    def test_clone_on_transport_upgrades_format_if_stacked_on_specified(self):
        child_branch, new_child_transport = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.controldir.clone_on_transport(new_child_transport, stacked_on=child_branch.controldir.root_transport.base)
        self.assertEqual(child_branch.controldir.root_transport.base, new_child.open_branch().get_stacked_on_url())
        repo = new_child.open_repository()
        self.assertTrue(repo._format.supports_external_lookups)
        self.assertFalse(repo.supports_rich_root())

    def test_sprout_upgrades_to_rich_root_format_if_needed(self):
        child_branch, new_child_transport = self.prepare_default_stacking(child_format='rich-root-pack')
        new_child = child_branch.controldir.sprout(new_child_transport.base, stacked=True)
        repo = new_child.open_repository()
        self.assertTrue(repo._format.supports_external_lookups)
        self.assertTrue(repo.supports_rich_root())

    def test_add_fallback_repo_handles_absolute_urls(self):
        stack_on = self.make_branch('stack_on', format='1.6')
        repo = self.make_repository('repo', format='1.6')
        policy = bzrdir.UseExistingRepository(repo, stack_on.base)
        policy._add_fallback(repo)

    def test_add_fallback_repo_handles_relative_urls(self):
        stack_on = self.make_branch('stack_on', format='1.6')
        repo = self.make_repository('repo', format='1.6')
        policy = bzrdir.UseExistingRepository(repo, '.', stack_on.base)
        policy._add_fallback(repo)

    def test_configure_relative_branch_stacking_url(self):
        stack_on = self.make_branch('stack_on', format='1.6')
        stacked = self.make_branch('stack_on/stacked', format='1.6')
        policy = bzrdir.UseExistingRepository(stacked.repository, '.', stack_on.base)
        policy.configure_branch(stacked)
        self.assertEqual('..', stacked.get_stacked_on_url())

    def test_relative_branch_stacking_to_absolute(self):
        stack_on = self.make_branch('stack_on', format='1.6')
        stacked = self.make_branch('stack_on/stacked', format='1.6')
        policy = bzrdir.UseExistingRepository(stacked.repository, '.', self.get_readonly_url('stack_on'))
        policy.configure_branch(stacked)
        self.assertEqual(self.get_readonly_url('stack_on'), stacked.get_stacked_on_url())