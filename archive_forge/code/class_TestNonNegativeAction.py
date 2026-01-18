import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
class TestNonNegativeAction(utils.TestCase):

    def setUp(self):
        super(TestNonNegativeAction, self).setUp()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--foo', metavar='<foo>', type=int, action=parseractions.NonNegativeAction)

    def test_negative_values(self):
        self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, '--foo -1'.split())

    def test_zero_values(self):
        results = self.parser.parse_args('--foo 0'.split())
        actual = getattr(results, 'foo', None)
        self.assertEqual(actual, 0)

    def test_positive_values(self):
        results = self.parser.parse_args('--foo 1'.split())
        actual = getattr(results, 'foo', None)
        self.assertEqual(actual, 1)