from breezy.tests import TestCaseInTempDir
class DemoTest(TestCaseInTempDir):

    def test_nothing(self):
        self.assertEqual(1, 1)