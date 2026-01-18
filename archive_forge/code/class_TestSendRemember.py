from typing import List
from breezy import branch, urlutils
from breezy.tests import script
class TestSendRemember(script.TestCaseWithTransportAndScript, TestRememberMixin):
    working_dir = 'work'
    command = ['send', '-o-']
    first_use_args = ['../parent', '../grand_parent']
    next_uses_args = ['../new_parent', '../new_grand_parent']

    def setUp(self):
        super().setUp()
        self.run_script("\n            $ brz init grand_parent\n            $ cd grand_parent\n            $ echo grand_parent > file\n            $ brz add\n            $ brz commit -m 'initial commit'\n            $ cd ..\n            $ brz branch grand_parent parent\n            $ cd parent\n            $ echo parent > file\n            $ brz commit -m 'parent'\n            $ cd ..\n            $ brz branch parent {working_dir}\n            $ cd {working_dir}\n            $ echo {working_dir} > file\n            $ brz commit -m '{working_dir}'\n            $ cd ..\n            ".format(working_dir=self.working_dir), null_output_matches_anything=True)

    def setup_next_uses(self):
        self.do_command(*self.first_use_args)
        self.run_script('\n            $ brz branch grand_parent new_grand_parent\n            $ brz branch parent new_parent\n            ', null_output_matches_anything=True)

    def assertLocations(self, expected_locations):
        if not expected_locations:
            expected_submit_branch, expected_public_branch = (None, None)
        else:
            expected_submit_branch, expected_public_branch = expected_locations
        br, _ = branch.Branch.open_containing(self.working_dir)
        self.assertEqual(expected_submit_branch, br.get_submit_branch())
        self.assertEqual(expected_public_branch, br.get_public_branch())