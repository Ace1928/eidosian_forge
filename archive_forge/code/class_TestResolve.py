from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
class TestResolve(script.TestCaseWithTransportAndScript):

    def setUp(self):
        super().setUp()
        test_conflicts.make_tree_with_conflicts(self, 'branch', 'other')

    def test_resolve_one_by_one(self):
        self.run_script('$ cd branch\n$ brz conflicts\nText conflict in my_other_file\nPath conflict: mydir3 / mydir2\nText conflict in myfile\n$ brz resolve myfile\n2>1 conflict resolved, 2 remaining\n$ brz resolve my_other_file\n2>1 conflict resolved, 1 remaining\n$ brz resolve mydir2\n2>1 conflict resolved, 0 remaining\n')

    def test_resolve_all(self):
        self.run_script('$ cd branch\n$ brz resolve --all\n2>3 conflicts resolved, 0 remaining\n$ brz conflicts\n')

    def test_resolve_from_subdir(self):
        self.run_script('$ mkdir branch/subdir\n$ cd branch/subdir\n$ brz resolve ../myfile\n2>1 conflict resolved, 2 remaining\n')

    def test_resolve_via_directory_option(self):
        self.run_script('$ brz resolve -d branch myfile\n2>1 conflict resolved, 2 remaining\n')

    def test_resolve_all_via_directory_option(self):
        self.run_script('$ brz resolve -d branch --all\n2>3 conflicts resolved, 0 remaining\n$ brz conflicts -d branch\n')

    def test_bug_842575_manual_rm(self):
        self.run_script('$ brz init -q trunk\n$ echo original > trunk/foo\n$ brz add -q trunk/foo\n$ brz commit -q -m first trunk\n$ brz checkout -q trunk tree\n$ brz rm -q trunk/foo\n$ brz commit -q -m second trunk\n$ echo modified > tree/foo\n$ brz update tree\n2>RM  foo => foo.THIS\n2>Contents conflict in foo\n2>1 conflicts encountered.\n2>Updated to revision 2 of branch ...\n$ rm tree/foo.BASE tree/foo.THIS\n$ brz resolve --all -d tree\n2>1 conflict resolved, 0 remaining\n')
        try:
            self.run_script('$ brz status tree\n')
        except AssertionError:
            raise KnownFailure('bug #842575')

    def test_bug_842575_take_other(self):
        self.run_script('$ brz init -q trunk\n$ echo original > trunk/foo\n$ brz add -q trunk/foo\n$ brz commit -q -m first trunk\n$ brz checkout -q --lightweight trunk tree\n$ brz rm -q trunk/foo\n$ brz ignore -d trunk foo\n$ brz commit -q -m second trunk\n$ echo modified > tree/foo\n$ brz update tree\n2>+N  .bzrignore\n2>RM  foo => foo.THIS\n2>Contents conflict in foo\n2>1 conflicts encountered.\n2>Updated to revision 2 of branch ...\n$ brz resolve --take-other --all -d tree\n2>1 conflict resolved, 0 remaining\n')
        try:
            self.run_script('$ brz status tree\n$ echo mustignore > tree/foo\n$ brz status tree\n')
        except AssertionError:
            raise KnownFailure('bug 842575')