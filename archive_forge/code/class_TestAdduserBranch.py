import os
from breezy import merge, osutils, tests
from breezy.plugins import po_merge
from breezy.tests import features, script
class TestAdduserBranch(script.TestCaseWithTransportAndScript):
    """Sanity checks on the adduser branch content."""

    def setUp(self):
        super().setUp()
        self.builder = make_adduser_branch(self, 'adduser')

    def assertAdduserBranchContent(self, revid):
        env = dict(revid=revid, branch_name=revid)
        self.run_script('$ brz branch adduser -rrevid:%(revid)s %(branch_name)s\n' % env, null_output_matches_anything=True)
        self.assertFileEqual(_Adduser['%(revid)s_pot' % env], '%(branch_name)s/po/adduser.pot' % env)
        self.assertFileEqual(_Adduser['%(revid)s_po' % env], '%(branch_name)s/po/fr.po' % env)

    def test_base(self):
        self.assertAdduserBranchContent('base')

    def test_this(self):
        self.assertAdduserBranchContent('this')

    def test_other(self):
        self.assertAdduserBranchContent('other')