from datetime import datetime
from ....tests import TestCase
from ..forge import parse_timestring
class ParseTimestringTests(TestCase):

    def test_simple(self):
        self.assertEqual(datetime(2011, 1, 26, 19, 1, 12), parse_timestring('2011-01-26T19:01:12Z'))