import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
class DeprecatedActionTest(utils.TestCase):

    @mock.patch.object(argparse.Action, '__init__', return_value=None)
    def test_init_emptyhelp_nouse(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertIsNone(result.use)
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', 'Deprecated', {'a': 1, 'b': 2, 'c': 3}))
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)

    @mock.patch.object(novaclient.shell.argparse.Action, '__init__', return_value=None)
    def test_init_emptyhelp_withuse(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', use='use this instead', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertEqual(result.use, 'use this instead')
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', 'Deprecated; use this instead', {'a': 1, 'b': 2, 'c': 3}))
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help='Deprecated; use this instead', a=1, b=2, c=3)

    @mock.patch.object(argparse.Action, '__init__', return_value=None)
    def test_init_withhelp_nouse(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', help='some help', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertIsNone(result.use)
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', 'some help (Deprecated)', {'a': 1, 'b': 2, 'c': 3}))
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help='some help (Deprecated)', a=1, b=2, c=3)

    @mock.patch.object(novaclient.shell.argparse.Action, '__init__', return_value=None)
    def test_init_withhelp_withuse(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', help='some help', use='use this instead', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertEqual(result.use, 'use this instead')
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', 'some help (Deprecated; use this instead)', {'a': 1, 'b': 2, 'c': 3}))
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help='some help (Deprecated; use this instead)', a=1, b=2, c=3)

    @mock.patch.object(argparse.Action, '__init__', return_value=None)
    def test_init_suppresshelp_nouse(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', help=argparse.SUPPRESS, a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertIsNone(result.use)
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', argparse.SUPPRESS, {'a': 1, 'b': 2, 'c': 3}))
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help=argparse.SUPPRESS, a=1, b=2, c=3)

    @mock.patch.object(novaclient.shell.argparse.Action, '__init__', return_value=None)
    def test_init_suppresshelp_withuse(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', help=argparse.SUPPRESS, use='use this instead', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertEqual(result.use, 'use this instead')
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', argparse.SUPPRESS, {'a': 1, 'b': 2, 'c': 3}))
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help=argparse.SUPPRESS, a=1, b=2, c=3)

    @mock.patch.object(argparse.Action, '__init__', return_value=None)
    def test_init_action_nothing(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='nothing', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertIsNone(result.use)
        self.assertIs(result.real_action_args, False)
        self.assertIsNone(result.real_action)
        mock_init.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)

    @mock.patch.object(argparse.Action, '__init__', return_value=None)
    def test_init_action_string(self, mock_init):
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='store', a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertIsNone(result.use)
        self.assertEqual(result.real_action_args, ('option_strings', 'dest', 'Deprecated', {'a': 1, 'b': 2, 'c': 3}))
        self.assertEqual(result.real_action, 'store')
        mock_init.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)

    @mock.patch.object(argparse.Action, '__init__', return_value=None)
    def test_init_action_other(self, mock_init):
        action = mock.Mock()
        result = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action=action, a=1, b=2, c=3)
        self.assertEqual(result.emitted, set())
        self.assertIsNone(result.use)
        self.assertIs(result.real_action_args, False)
        self.assertEqual(result.real_action, action.return_value)
        mock_init.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)
        action.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)

    @mock.patch.object(sys, 'stderr', io.StringIO())
    def test_get_action_nolookup(self):
        action_class = mock.Mock()
        parser = mock.Mock(**{'_registry_get.return_value': action_class})
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='nothing', const=1)
        obj.real_action = 'action'
        result = obj._get_action(parser)
        self.assertEqual(result, 'action')
        self.assertEqual(obj.real_action, 'action')
        self.assertFalse(parser._registry_get.called)
        self.assertFalse(action_class.called)
        self.assertEqual(sys.stderr.getvalue(), '')

    @mock.patch.object(sys, 'stderr', io.StringIO())
    def test_get_action_lookup_noresult(self):
        parser = mock.Mock(**{'_registry_get.return_value': None})
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='store', const=1)
        result = obj._get_action(parser)
        self.assertIsNone(result)
        self.assertIsNone(obj.real_action)
        parser._registry_get.assert_called_once_with('action', 'store')
        self.assertEqual(sys.stderr.getvalue(), 'WARNING: Programming error: Unknown real action "store"\n')

    @mock.patch.object(sys, 'stderr', io.StringIO())
    def test_get_action_lookup_withresult(self):
        action_class = mock.Mock()
        parser = mock.Mock(**{'_registry_get.return_value': action_class})
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='store', const=1)
        result = obj._get_action(parser)
        self.assertEqual(result, action_class.return_value)
        self.assertEqual(obj.real_action, action_class.return_value)
        parser._registry_get.assert_called_once_with('action', 'store')
        action_class.assert_called_once_with('option_strings', 'dest', help='Deprecated', const=1)
        self.assertEqual(sys.stderr.getvalue(), '')

    @mock.patch.object(sys, 'stderr', io.StringIO())
    @mock.patch.object(novaclient.shell.DeprecatedAction, '_get_action')
    def test_call_unemitted_nouse(self, mock_get_action):
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest')
        obj('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(obj.emitted, set(['option_string']))
        mock_get_action.assert_called_once_with('parser')
        mock_get_action.return_value.assert_called_once_with('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(sys.stderr.getvalue(), 'WARNING: Option "option_string" is deprecated\n')

    @mock.patch.object(sys, 'stderr', io.StringIO())
    @mock.patch.object(novaclient.shell.DeprecatedAction, '_get_action')
    def test_call_unemitted_withuse(self, mock_get_action):
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', use='use this instead')
        obj('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(obj.emitted, set(['option_string']))
        mock_get_action.assert_called_once_with('parser')
        mock_get_action.return_value.assert_called_once_with('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(sys.stderr.getvalue(), 'WARNING: Option "option_string" is deprecated; use this instead\n')

    @mock.patch.object(sys, 'stderr', io.StringIO())
    @mock.patch.object(novaclient.shell.DeprecatedAction, '_get_action')
    def test_call_emitted_nouse(self, mock_get_action):
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest')
        obj.emitted.add('option_string')
        obj('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(obj.emitted, set(['option_string']))
        mock_get_action.assert_called_once_with('parser')
        mock_get_action.return_value.assert_called_once_with('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(sys.stderr.getvalue(), '')

    @mock.patch.object(sys, 'stderr', io.StringIO())
    @mock.patch.object(novaclient.shell.DeprecatedAction, '_get_action')
    def test_call_emitted_withuse(self, mock_get_action):
        obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', use='use this instead')
        obj.emitted.add('option_string')
        obj('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(obj.emitted, set(['option_string']))
        mock_get_action.assert_called_once_with('parser')
        mock_get_action.return_value.assert_called_once_with('parser', 'namespace', 'values', 'option_string')
        self.assertEqual(sys.stderr.getvalue(), '')