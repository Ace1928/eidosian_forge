import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
class TestMultiKeyValueCommaAction(utils.TestCase):

    def setUp(self):
        super(TestMultiKeyValueCommaAction, self).setUp()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--test', metavar='req1=xxx,yyy', action=parseractions.MultiKeyValueCommaAction, dest='test', default=None, required_keys=['req1'], optional_keys=['opt2'], help='Test')

    def test_mkvca_required(self):
        results = self.parser.parse_args(['--test', 'req1=aaa,bbb'])
        actual = getattr(results, 'test', [])
        expect = [{'req1': 'aaa,bbb'}]
        self.assertCountEqual(expect, actual)
        results = self.parser.parse_args(['--test', 'req1='])
        actual = getattr(results, 'test', [])
        expect = [{'req1': ''}]
        self.assertCountEqual(expect, actual)
        results = self.parser.parse_args(['--test', 'req1=aaa,bbb', '--test', 'req1='])
        actual = getattr(results, 'test', [])
        expect = [{'req1': 'aaa,bbb'}, {'req1': ''}]
        self.assertCountEqual(expect, actual)

    def test_mkvca_optional(self):
        results = self.parser.parse_args(['--test', 'req1=aaa,bbb'])
        actual = getattr(results, 'test', [])
        expect = [{'req1': 'aaa,bbb'}]
        self.assertCountEqual(expect, actual)
        results = self.parser.parse_args(['--test', 'req1=aaa,bbb', '--test', 'req1=,opt2=ccc'])
        actual = getattr(results, 'test', [])
        expect = [{'req1': 'aaa,bbb'}, {'req1': '', 'opt2': 'ccc'}]
        self.assertCountEqual(expect, actual)
        try:
            results = self.parser.parse_args(['--test', 'req1=aaa,bbb', '--test', 'opt2=ccc'])
            self.fail('ArgumentTypeError should be raised')
        except argparse.ArgumentTypeError as e:
            self.assertEqual('Missing required keys req1.\nRequired keys are: req1', str(e))

    def test_mkvca_multiples(self):
        results = self.parser.parse_args(['--test', 'req1=aaa,bbb,opt2=ccc'])
        actual = getattr(results, 'test', [])
        expect = [{'req1': 'aaa,bbb', 'opt2': 'ccc'}]
        self.assertCountEqual(expect, actual)

    def test_mkvca_no_required_optional(self):
        self.parser.add_argument('--test-empty', metavar='req1=xxx,yyy', action=parseractions.MultiKeyValueCommaAction, dest='test_empty', default=None, required_keys=[], optional_keys=[], help='Test')
        results = self.parser.parse_args(['--test-empty', 'req1=aaa,bbb'])
        actual = getattr(results, 'test_empty', [])
        expect = [{'req1': 'aaa,bbb'}]
        self.assertCountEqual(expect, actual)
        results = self.parser.parse_args(['--test-empty', 'xyz=aaa,bbb'])
        actual = getattr(results, 'test_empty', [])
        expect = [{'xyz': 'aaa,bbb'}]
        self.assertCountEqual(expect, actual)

    def test_mkvca_invalid_key(self):
        try:
            self.parser.parse_args(['--test', 'req1=aaa,bbb='])
            self.fail('ArgumentTypeError should be raised')
        except argparse.ArgumentTypeError as e:
            self.assertIn('Invalid keys bbb specified.\nValid keys are:', str(e))
        try:
            self.parser.parse_args(['--test', 'nnn=aaa'])
            self.fail('ArgumentTypeError should be raised')
        except argparse.ArgumentTypeError as e:
            self.assertIn('Invalid keys nnn specified.\nValid keys are:', str(e))

    def test_mkvca_value_no_key(self):
        try:
            self.parser.parse_args(['--test', 'req1=aaa,=bbb'])
            self.fail('ArgumentTypeError should be raised')
        except argparse.ArgumentTypeError as e:
            self.assertEqual("A key must be specified before '=': =bbb", str(e))
        try:
            self.parser.parse_args(['--test', '=nnn'])
            self.fail('ArgumentTypeError should be raised')
        except argparse.ArgumentTypeError as e:
            self.assertEqual("A key must be specified before '=': =nnn", str(e))
        try:
            self.parser.parse_args(['--test', 'nnn'])
            self.fail('ArgumentTypeError should be raised')
        except argparse.ArgumentTypeError as e:
            self.assertIn('A key=value pair is required:', str(e))

    def test_mkvca_required_keys_not_list(self):
        self.assertRaises(TypeError, self.parser.add_argument, '--test-required-dict', metavar='req1=xxx', action=parseractions.MultiKeyValueCommaAction, dest='test_required_dict', default=None, required_keys={'aaa': 'bbb'}, optional_keys=['opt1', 'opt2'], help='Test')

    def test_mkvca_optional_keys_not_list(self):
        self.assertRaises(TypeError, self.parser.add_argument, '--test-optional-dict', metavar='req1=xxx', action=parseractions.MultiKeyValueCommaAction, dest='test_optional_dict', default=None, required_keys=['req1', 'req2'], optional_keys={'aaa': 'bbb'}, help='Test')