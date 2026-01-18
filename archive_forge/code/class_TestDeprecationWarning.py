import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
class TestDeprecationWarning(tests.TestCaseWithTransport):
    """The deprecation warning is controlled via a global variable:
    repository._deprecation_warning_done. As such, it can be emitted only once
    during a brz invocation, no matter how many repositories are involved.

    It would be better if it was a repo attribute instead but that's far more
    work than I want to do right now -- vila 20091215.
    """

    def setUp(self):
        super().setUp()
        self.addCleanup(repository.format_registry.remove, TestObsoleteRepoFormat)
        repository.format_registry.register(TestObsoleteRepoFormat)
        self.addCleanup(controldir.format_registry.remove, 'testobsolete')
        bzr.register_metadir(controldir.format_registry, 'testobsolete', 'breezy.tests.blackbox.test_exceptions.TestObsoleteRepoFormat', branch_format='breezy.bzr.branch.BzrBranchFormat7', tree_format='breezy.bzr.workingtree_4.WorkingTreeFormat6', deprecated=True, help='Same as 2a, but with an obsolete repo format.')
        self.disable_deprecation_warning()

    def enable_deprecation_warning(self, repo=None):
        """repo is not used yet since _deprecation_warning_done is a global"""
        repository._deprecation_warning_done = False

    def disable_deprecation_warning(self, repo=None):
        """repo is not used yet since _deprecation_warning_done is a global"""
        repository._deprecation_warning_done = True

    def make_obsolete_repo(self, path):
        format = controldir.format_registry.make_controldir('testobsolete')
        tree = self.make_branch_and_tree(path, format=format)
        return tree

    def check_warning(self, present):
        if present:
            check = self.assertContainsRe
        else:
            check = self.assertNotContainsRe
        check(self.get_log(), 'WARNING.*brz upgrade')

    def test_repository_deprecation_warning(self):
        """Old formats give a warning"""
        self.make_obsolete_repo('foo')
        self.enable_deprecation_warning()
        out, err = self.run_bzr('status', working_dir='foo')
        self.check_warning(True)

    def test_repository_deprecation_warning_suppressed_global(self):
        """Old formats give a warning"""
        conf = config.GlobalStack()
        conf.set('suppress_warnings', 'format_deprecation')
        self.make_obsolete_repo('foo')
        self.enable_deprecation_warning()
        out, err = self.run_bzr('status', working_dir='foo')
        self.check_warning(False)

    def test_repository_deprecation_warning_suppressed_locations(self):
        """Old formats give a warning"""
        self.make_obsolete_repo('foo')
        conf = config.LocationStack(osutils.pathjoin(self.test_dir, 'foo'))
        conf.set('suppress_warnings', 'format_deprecation')
        self.enable_deprecation_warning()
        out, err = self.run_bzr('status', working_dir='foo')
        self.check_warning(False)

    def test_repository_deprecation_warning_suppressed_branch(self):
        """Old formats give a warning"""
        tree = self.make_obsolete_repo('foo')
        conf = tree.branch.get_config_stack()
        conf.set('suppress_warnings', 'format_deprecation')
        self.enable_deprecation_warning()
        out, err = self.run_bzr('status', working_dir='foo')
        self.check_warning(False)