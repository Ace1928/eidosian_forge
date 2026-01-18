from a single quote by the algorithm. Therefore, a text like::
import re, sys
class TestSmartypantsAllAttributes(unittest.TestCase):

    def test_dates(self):
        self.assertEqual(smartyPants("1440-80's"), '1440-80’s')
        self.assertEqual(smartyPants("1440-'80s"), '1440-’80s')
        self.assertEqual(smartyPants("1440---'80s"), '1440–’80s')
        self.assertEqual(smartyPants("1960's"), '1960’s')
        self.assertEqual(smartyPants("one two '60s"), 'one two ’60s')
        self.assertEqual(smartyPants("'60s"), '’60s')

    def test_educated_quotes(self):
        self.assertEqual(smartyPants('"Isn\'t this fun?"'), '“Isn’t this fun?”')

    def test_html_tags(self):
        text = '<a src="foo">more</a>'
        self.assertEqual(smartyPants(text), text)