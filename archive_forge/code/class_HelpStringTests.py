from twisted.python import usage
from twisted.trial import unittest
class HelpStringTests(unittest.TestCase):
    """
    Test generated help strings.
    """

    def setUp(self):
        """
        Instantiate a well-behaved Options class.
        """
        self.niceArgV = '--long Alpha -n Beta --shortless Gamma -f --myflag --myparam Tofu'.split()
        self.nice = WellBehaved()

    def test_noGoBoom(self):
        """
        __str__ shouldn't go boom.
        """
        try:
            self.nice.__str__()
        except Exception as e:
            self.fail(e)

    def test_whitespaceStripFlagsAndParameters(self):
        """
        Extra whitespace in flag and parameters docs is stripped.
        """
        lines = [s for s in str(self.nice).splitlines() if s.find('aflag') >= 0]
        self.assertTrue(len(lines) > 0)
        self.assertTrue(lines[0].find('flagallicious') >= 0)