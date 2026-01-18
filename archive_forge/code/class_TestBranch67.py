from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class TestBranch67:
    """Common tests for both branch 6 and 7 which are mostly the same."""

    def get_format_name(self):
        raise NotImplementedError(self.get_format_name)

    def get_format_name_subtree(self):
        raise NotImplementedError(self.get_format_name)

    def get_class(self):
        raise NotImplementedError(self.get_class)

    def test_creation(self):
        format = bzrdir.BzrDirMetaFormat1()
        format.set_branch_format(_mod_bzrbranch.BzrBranchFormat6())
        branch = self.make_branch('a', format=format)
        self.assertIsInstance(branch, self.get_class())
        branch = self.make_branch('b', format=self.get_format_name())
        self.assertIsInstance(branch, self.get_class())
        branch = _mod_branch.Branch.open('a')
        self.assertIsInstance(branch, self.get_class())

    def test_layout(self):
        branch = self.make_branch('a', format=self.get_format_name())
        self.assertPathExists('a/.bzr/branch/last-revision')
        self.assertPathDoesNotExist('a/.bzr/branch/revision-history')
        self.assertPathDoesNotExist('a/.bzr/branch/references')

    def test_config(self):
        """Ensure that all configuration data is stored in the branch"""
        branch = self.make_branch('a', format=self.get_format_name())
        branch.set_parent('http://example.com')
        self.assertPathDoesNotExist('a/.bzr/branch/parent')
        self.assertEqual('http://example.com', branch.get_parent())
        branch.set_push_location('sftp://example.com')
        conf = branch.get_config_stack()
        self.assertEqual('sftp://example.com', conf.get('push_location'))
        branch.set_bound_location('ftp://example.com')
        self.assertPathDoesNotExist('a/.bzr/branch/bound')
        self.assertEqual('ftp://example.com', branch.get_bound_location())

    def do_checkout_test(self, lightweight):
        tree = self.make_branch_and_tree('source', format=self.get_format_name_subtree())
        subtree = self.make_branch_and_tree('source/subtree', format=self.get_format_name_subtree())
        subsubtree = self.make_branch_and_tree('source/subtree/subsubtree', format=self.get_format_name_subtree())
        self.build_tree(['source/subtree/file', 'source/subtree/subsubtree/file'])
        subsubtree.add('file')
        subtree.add('file')
        subtree.add_reference(subsubtree)
        subtree.set_reference_info('subsubtree', subsubtree.branch.user_url)
        tree.add_reference(subtree)
        tree.set_reference_info('subtree', subtree.branch.user_url)
        tree.commit('a revision')
        subtree.commit('a subtree file')
        subsubtree.commit('a subsubtree file')
        tree.branch.create_checkout('target', lightweight=lightweight)
        self.assertPathExists('target')
        self.assertPathExists('target/subtree')
        self.assertPathExists('target/subtree/file')
        self.assertPathExists('target/subtree/subsubtree/file')
        subbranch = _mod_branch.Branch.open('target/subtree/subsubtree')
        if lightweight:
            self.assertEndsWith(subbranch.base, 'source/subtree/subsubtree/')
        else:
            self.assertEndsWith(subbranch.base, 'target/subtree/subsubtree/')

    def test_checkout_with_references(self):
        self.do_checkout_test(lightweight=False)

    def test_light_checkout_with_references(self):
        self.do_checkout_test(lightweight=True)