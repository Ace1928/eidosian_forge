import copy
from unittest import mock
from keystoneauth1 import session
from oslo_utils import uuidutils
import novaclient.api_versions
import novaclient.client
import novaclient.extension
from novaclient.tests.unit import utils
import novaclient.v2.client
class ClientsUtilsTest(utils.TestCase):

    @mock.patch('novaclient.client._discover_via_entry_points')
    @mock.patch('novaclient.client._discover_via_python_path')
    @mock.patch('novaclient.extension.Extension')
    def test_discover_extensions_all(self, mock_extension, mock_discover_via_python_path, mock_discover_via_entry_points):

        def make_gen(start, end):

            def f(*args, **kwargs):
                for i in range(start, end):
                    yield ('name-%s' % i, i)
            return f
        mock_discover_via_python_path.side_effect = make_gen(0, 3)
        mock_discover_via_entry_points.side_effect = make_gen(3, 4)
        version = novaclient.api_versions.APIVersion('2.0')
        result = novaclient.client.discover_extensions(version)
        self.assertEqual([mock.call('name-%s' % i, i) for i in range(0, 4)], mock_extension.call_args_list)
        mock_discover_via_python_path.assert_called_once_with()
        mock_discover_via_entry_points.assert_called_once_with()
        self.assertEqual([mock_extension()] * 4, result)

    @mock.patch('novaclient.client.warnings')
    def test__check_arguments(self, mock_warnings):
        release = 'Coolest'
        novaclient.client._check_arguments({}, release=release, deprecated_name='foo')
        self.assertFalse(mock_warnings.warn.called)
        novaclient.client._check_arguments({}, release=release, deprecated_name='foo', right_name='bar')
        self.assertFalse(mock_warnings.warn.called)
        original_kwargs = {'foo': 'text'}
        actual_kwargs = copy.copy(original_kwargs)
        self.assertEqual(original_kwargs, actual_kwargs)
        novaclient.client._check_arguments(actual_kwargs, release=release, deprecated_name='foo', right_name='bar')
        self.assertNotEqual(original_kwargs, actual_kwargs)
        self.assertEqual({'bar': original_kwargs['foo']}, actual_kwargs)
        self.assertTrue(mock_warnings.warn.called)
        mock_warnings.warn.reset_mock()
        original_kwargs = {'foo': 'text'}
        actual_kwargs = copy.copy(original_kwargs)
        self.assertEqual(original_kwargs, actual_kwargs)
        novaclient.client._check_arguments(actual_kwargs, release=release, deprecated_name='foo')
        self.assertNotEqual(original_kwargs, actual_kwargs)
        self.assertEqual({}, actual_kwargs)
        self.assertTrue(mock_warnings.warn.called)