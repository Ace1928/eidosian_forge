from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
class AutomaticTagNameTests(per_branch.TestCaseWithBranch):

    def setUp(self):
        super().setUp()
        if isinstance(self.branch_format, bzrbranch.BranchReferenceFormat):
            raise tests.TestSkipped("BranchBuilder can't make reference branches.")
        self.builder = self.make_branch_builder('.')
        self.builder.build_snapshot(None, [('add', ('', None, 'directory', None))], message='foo')
        self.branch = self.builder.get_branch()
        if not self.branch._format.supports_tags():
            raise tests.TestSkipped("format %s doesn't support tags" % self.branch._format)

    def test_no_functions(self):
        rev = self.branch.last_revision()
        self.assertEqual(None, self.branch.automatic_tag_name(rev))

    def test_returns_tag_name(self):

        def get_tag_name(br, revid):
            return 'foo'
        branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name, 'get tag name foo')
        self.assertEqual('foo', self.branch.automatic_tag_name(self.branch.last_revision()))

    def test_uses_first_return(self):

        def get_tag_name_1(br, revid):
            return 'foo1'

        def get_tag_name_2(br, revid):
            return 'foo2'
        branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_1, 'tagname1')
        branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_2, 'tagname2')
        self.assertEqual('foo1', self.branch.automatic_tag_name(self.branch.last_revision()))

    def test_ignores_none(self):

        def get_tag_name_1(br, revid):
            return None

        def get_tag_name_2(br, revid):
            return 'foo2'
        branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_1, 'tagname1')
        branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_2, 'tagname2')
        self.assertEqual('foo2', self.branch.automatic_tag_name(self.branch.last_revision()))