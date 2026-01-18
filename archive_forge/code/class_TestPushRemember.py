from typing import List
from breezy import branch, urlutils
from breezy.tests import script
class TestPushRemember(script.TestCaseWithTransportAndScript, TestRememberMixin):
    working_dir = 'work'
    command = ['push']
    first_use_args = ['../target']
    next_uses_args = ['../new_target']

    def setUp(self):
        super().setUp()
        self.run_script("\n            $ brz init {working_dir}\n            $ cd {working_dir}\n            $ echo some content > file\n            $ brz add\n            $ brz commit -m 'initial commit'\n            $ cd ..\n            ".format(working_dir=self.working_dir), null_output_matches_anything=True)

    def setup_next_uses(self):
        self.do_command(*self.first_use_args)
        self.run_script("\n            $ cd {working_dir}\n            $ echo new content > file\n            $ brz commit -m 'new content'\n            $ cd ..\n            ".format(working_dir=self.working_dir), null_output_matches_anything=True)

    def assertLocations(self, expected_locations):
        br, _ = branch.Branch.open_containing(self.working_dir)
        if not expected_locations:
            self.assertEqual(None, br.get_push_location())
        else:
            expected_push_location = expected_locations[0]
            push_location = urlutils.relative_url(br.base, br.get_push_location())
            self.assertIsSameRealPath(expected_push_location, push_location)