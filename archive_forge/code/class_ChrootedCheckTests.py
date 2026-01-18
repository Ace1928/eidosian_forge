from breezy.tests import ChrootedTestCase, TestCaseWithTransport
class ChrootedCheckTests(ChrootedTestCase):

    def test_check_missing_branch(self):
        out, err = self.run_bzr('check --branch %s' % self.get_readonly_url(''))
        self.assertEqual(err, 'No branch found at specified location.\n')

    def test_check_missing_repository(self):
        out, err = self.run_bzr('check --repo %s' % self.get_readonly_url(''))
        self.assertEqual(err, 'No repository found at specified location.\n')

    def test_check_missing_everything(self):
        out, err = self.run_bzr('check %s' % self.get_readonly_url(''))
        self.assertEqual(err, 'No working tree found at specified location.\nNo branch found at specified location.\nNo repository found at specified location.\n')