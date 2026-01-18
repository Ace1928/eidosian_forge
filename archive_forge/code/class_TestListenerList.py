import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestListenerList(TestListener):

    def setUp(self):
        super().setUp()
        self.datalist = (tuple((attr_consts.LISTENER_ATTRS[k] for k in self.columns)),)
        self.cmd = listener.ListListener(self.app, None)

    def test_listener_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.listener_list.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_listener_list_with_options(self):
        arglist = ['--name', 'rainbarrel']
        verifylist = [('name', 'rainbarrel')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.listener_list.assert_called_with(name='rainbarrel')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_listener_list_with_tags(self):
        arglist = ['--tags', 'foo,bar']
        verifylist = [('tags', ['foo', 'bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.listener_list.assert_called_with(tags=['foo', 'bar'])
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_listener_list_with_any_tags(self):
        arglist = ['--any-tags', 'foo,bar']
        verifylist = [('any_tags', ['foo', 'bar'])]
        expected_attrs = {'tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.listener_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_listener_list_with_not_tags(self):
        arglist = ['--not-tags', 'foo,bar']
        verifylist = [('not_tags', ['foo', 'bar'])]
        expected_attrs = {'not-tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.listener_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_listener_list_with_not_any_tags(self):
        arglist = ['--not-any-tags', 'foo,bar']
        verifylist = [('not_any_tags', ['foo', 'bar'])]
        expected_attrs = {'not-tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.listener_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))