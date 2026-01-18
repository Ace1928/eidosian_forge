from breezy import branch, tests
def _check_revision_history(self, location='', working_dir=None):
    rh = self.run_bzr(['revision-history', location], working_dir=working_dir)[0]
    self.assertEqual(rh, 'revision_1\nrevision_2\nrevision_3\n')