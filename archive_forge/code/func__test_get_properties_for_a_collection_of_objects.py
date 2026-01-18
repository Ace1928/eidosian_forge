import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.get_prop_spec')
@mock.patch('oslo_vmware.vim_util.get_obj_spec')
@mock.patch('oslo_vmware.vim_util.get_prop_filter_spec')
def _test_get_properties_for_a_collection_of_objects(self, objs, max_objects, mock_get_prop_filter_spec, mock_get_obj_spec, mock_get_prop_spec):
    vim = mock.Mock()
    if len(objs) == 0:
        self.assertEqual([], vim_util.get_properties_for_a_collection_of_objects(vim, 'VirtualMachine', [], {}))
        return
    mock_prop_spec = mock.Mock()
    mock_get_prop_spec.return_value = mock_prop_spec
    mock_get_obj_spec.side_effect = [mock.Mock() for obj in objs]
    get_obj_spec_calls = [mock.call(vim.client.factory, obj) for obj in objs]
    mock_prop_spec = mock.Mock()
    mock_get_prop_spec.return_value = mock_prop_spec
    mock_prop_filter_spec = mock.Mock()
    mock_get_prop_filter_spec.return_value = mock_prop_filter_spec
    mock_options = mock.Mock()
    vim.client.factory.create.return_value = mock_options
    mock_return_value = mock.Mock()
    vim.RetrievePropertiesEx.return_value = mock_return_value
    res = vim_util.get_properties_for_a_collection_of_objects(vim, 'VirtualMachine', objs, ['runtime'], max_objects)
    self.assertEqual(mock_return_value, res)
    mock_get_prop_spec.assert_called_once_with(vim.client.factory, 'VirtualMachine', ['runtime'])
    self.assertEqual(get_obj_spec_calls, mock_get_obj_spec.mock_calls)
    vim.client.factory.create.assert_called_once_with('ns0:RetrieveOptions')
    self.assertEqual(max_objects if max_objects else len(objs), mock_options.maxObjects)
    vim.RetrievePropertiesEx.assert_called_once_with(vim.service_content.propertyCollector, specSet=[mock_prop_filter_spec], options=mock_options)