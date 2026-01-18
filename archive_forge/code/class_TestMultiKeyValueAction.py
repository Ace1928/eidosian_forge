import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
class TestMultiKeyValueAction(utils.TestCase):

    def setUp(self):
        super(TestMultiKeyValueAction, self).setUp()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--test', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test', default=None, required_keys=['req1', 'req2'], optional_keys=['opt1', 'opt2'], help='Test')

    def test_good_values(self):
        results = self.parser.parse_args(['--test', 'req1=aaa,req2=bbb', '--test', 'req1=,req2='])
        actual = getattr(results, 'test', [])
        expect = [{'req1': 'aaa', 'req2': 'bbb'}, {'req1': '', 'req2': ''}]
        self.assertCountEqual(expect, actual)

    def test_empty_required_optional(self):
        self.parser.add_argument('--test-empty', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test_empty', default=None, required_keys=[], optional_keys=[], help='Test')
        results = self.parser.parse_args(['--test-empty', 'req1=aaa,req2=bbb', '--test-empty', 'req1=,req2='])
        actual = getattr(results, 'test_empty', [])
        expect = [{'req1': 'aaa', 'req2': 'bbb'}, {'req1': '', 'req2': ''}]
        self.assertCountEqual(expect, actual)

    def test_error_values_with_comma(self):
        data_list = [['--test', 'mmm,nnn=zzz'], ['--test', 'nnn=zzz,='], ['--test', 'nnn=zzz,=zzz']]
        for data in data_list:
            self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, data)

    def test_error_values_without_comma(self):
        self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, ['--test', 'mmmnnn'])

    def test_missing_key(self):
        self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, ['--test', 'req2=ddd'])

    def test_invalid_key(self):
        self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, ['--test', 'req1=aaa,req2=bbb,aaa=req1'])

    def test_required_keys_not_list(self):
        self.assertRaises(TypeError, self.parser.add_argument, '--test-required-dict', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test_required_dict', default=None, required_keys={'aaa': 'bbb'}, optional_keys=['opt1', 'opt2'], help='Test')

    def test_optional_keys_not_list(self):
        self.assertRaises(TypeError, self.parser.add_argument, '--test-optional-dict', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test_optional_dict', default=None, required_keys=['req1', 'req2'], optional_keys={'aaa': 'bbb'}, help='Test')