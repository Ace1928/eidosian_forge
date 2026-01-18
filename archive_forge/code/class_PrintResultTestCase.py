import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
class PrintResultTestCase(test_utils.TestCase):

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_dict(self):
        dict = {'key': 'value'}
        utils.print_dict(dict)
        self.assertEqual('+----------+-------+\n| Property | Value |\n+----------+-------+\n| key      | value |\n+----------+-------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_dict_wrap(self):
        dict = {'key1': 'not wrapped', 'key2': 'this will be wrapped'}
        utils.print_dict(dict, wrap=16)
        self.assertEqual('+----------+--------------+\n| Property | Value        |\n+----------+--------------+\n| key1     | not wrapped  |\n| key2     | this will be |\n|          | wrapped      |\n+----------+--------------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_list_sort_by_str(self):
        objs = [_FakeResult('k1', 1), _FakeResult('k3', 2), _FakeResult('k2', 3)]
        utils.print_list(objs, ['Name', 'Value'], sortby_index=0)
        self.assertEqual('+------+-------+\n| Name | Value |\n+------+-------+\n| k1   | 1     |\n| k2   | 3     |\n| k3   | 2     |\n+------+-------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_list_sort_by_integer(self):
        objs = [_FakeResult('k1', 1), _FakeResult('k3', 2), _FakeResult('k2', 3)]
        utils.print_list(objs, ['Name', 'Value'], sortby_index=1)
        self.assertEqual('+------+-------+\n| Name | Value |\n+------+-------+\n| k1   | 1     |\n| k3   | 2     |\n| k2   | 3     |\n+------+-------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_unicode_list(self):
        objs = [_FakeResult('k', '…')]
        utils.print_list(objs, ['Name', 'Value'])
        s = '…'
        self.assertEqual('+------+-------+\n| Name | Value |\n+------+-------+\n| k    | %s     |\n+------+-------+\n' % s, sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_list_sort_by_none(self):
        objs = [_FakeResult('k1', 1), _FakeResult('k3', 3), _FakeResult('k2', 2)]
        utils.print_list(objs, ['Name', 'Value'], sortby_index=None)
        self.assertEqual('+------+-------+\n| Name | Value |\n+------+-------+\n| k1   | 1     |\n| k3   | 3     |\n| k2   | 2     |\n+------+-------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_dict_dictionary(self):
        dict = {'k': {'foo': 'bar'}}
        utils.print_dict(dict)
        self.assertEqual('+----------+----------------+\n| Property | Value          |\n+----------+----------------+\n| k        | {"foo": "bar"} |\n+----------+----------------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_dict_list_dictionary(self):
        dict = {'k': [{'foo': 'bar'}]}
        utils.print_dict(dict)
        self.assertEqual('+----------+------------------+\n| Property | Value            |\n+----------+------------------+\n| k        | [{"foo": "bar"}] |\n+----------+------------------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_dict_list(self):
        dict = {'k': ['foo', 'bar']}
        utils.print_dict(dict)
        self.assertEqual('+----------+----------------+\n| Property | Value          |\n+----------+----------------+\n| k        | ["foo", "bar"] |\n+----------+----------------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_large_dict_list(self):
        dict = {'k': ['foo1', 'bar1', 'foo2', 'bar2', 'foo3', 'bar3', 'foo4', 'bar4']}
        utils.print_dict(dict, wrap=40)
        self.assertEqual('+----------+------------------------------------------+\n| Property | Value                                    |\n+----------+------------------------------------------+\n| k        | ["foo1", "bar1", "foo2", "bar2", "foo3", |\n|          | "bar3", "foo4", "bar4"]                  |\n+----------+------------------------------------------+\n', sys.stdout.getvalue())

    @mock.patch('sys.stdout', io.StringIO())
    def test_print_unicode_dict(self):
        dict = {'k': '…'}
        utils.print_dict(dict)
        s = '…'
        self.assertEqual('+----------+-------+\n| Property | Value |\n+----------+-------+\n| k        | %s     |\n+----------+-------+\n' % s, sys.stdout.getvalue())