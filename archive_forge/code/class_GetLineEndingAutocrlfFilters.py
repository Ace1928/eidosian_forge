from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
class GetLineEndingAutocrlfFilters(TestCase):

    def test_get_checkin_filter_autocrlf_default(self):
        checkin_filter = get_checkin_filter_autocrlf(b'false')
        self.assertEqual(checkin_filter, None)

    def test_get_checkin_filter_autocrlf_true(self):
        checkin_filter = get_checkin_filter_autocrlf(b'true')
        self.assertEqual(checkin_filter, convert_crlf_to_lf)

    def test_get_checkin_filter_autocrlf_input(self):
        checkin_filter = get_checkin_filter_autocrlf(b'input')
        self.assertEqual(checkin_filter, convert_crlf_to_lf)

    def test_get_checkout_filter_autocrlf_default(self):
        checkout_filter = get_checkout_filter_autocrlf(b'false')
        self.assertEqual(checkout_filter, None)

    def test_get_checkout_filter_autocrlf_true(self):
        checkout_filter = get_checkout_filter_autocrlf(b'true')
        self.assertEqual(checkout_filter, convert_lf_to_crlf)

    def test_get_checkout_filter_autocrlf_input(self):
        checkout_filter = get_checkout_filter_autocrlf(b'input')
        self.assertEqual(checkout_filter, None)