import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import object as obj
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
@mock.patch('openstackclient.api.object_store_v1.APIv1.object_list')
class TestObjectList(TestObject):
    columns = ('Name',)
    datalist = ((object_fakes.object_name_2,),)

    def setUp(self):
        super(TestObjectList, self).setUp()
        self.cmd = obj.ListObject(self.app, None)

    def test_object_list_objects_no_options(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT), copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = [object_fakes.container_name]
        verifylist = [('container', object_fakes.container_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        o_mock.assert_called_with(container=object_fakes.container_name)
        self.assertEqual(self.columns, columns)
        datalist = ((object_fakes.object_name_1,), (object_fakes.object_name_2,))
        self.assertEqual(datalist, tuple(data))

    def test_object_list_objects_prefix(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--prefix', 'floppy', object_fakes.container_name_2]
        verifylist = [('prefix', 'floppy'), ('container', object_fakes.container_name_2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'prefix': 'floppy'}
        o_mock.assert_called_with(container=object_fakes.container_name_2, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_object_list_objects_delimiter(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--delimiter', '=', object_fakes.container_name_2]
        verifylist = [('delimiter', '='), ('container', object_fakes.container_name_2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'delimiter': '='}
        o_mock.assert_called_with(container=object_fakes.container_name_2, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_object_list_objects_marker(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--marker', object_fakes.object_name_2, object_fakes.container_name_2]
        verifylist = [('marker', object_fakes.object_name_2), ('container', object_fakes.container_name_2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': object_fakes.object_name_2}
        o_mock.assert_called_with(container=object_fakes.container_name_2, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_object_list_objects_end_marker(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--end-marker', object_fakes.object_name_2, object_fakes.container_name_2]
        verifylist = [('end_marker', object_fakes.object_name_2), ('container', object_fakes.container_name_2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'end_marker': object_fakes.object_name_2}
        o_mock.assert_called_with(container=object_fakes.container_name_2, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_object_list_objects_limit(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--limit', '2', object_fakes.container_name_2]
        verifylist = [('limit', 2), ('container', object_fakes.container_name_2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'limit': 2}
        o_mock.assert_called_with(container=object_fakes.container_name_2, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_object_list_objects_long(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT), copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--long', object_fakes.container_name]
        verifylist = [('long', True), ('container', object_fakes.container_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {}
        o_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
        collist = ('Name', 'Bytes', 'Hash', 'Content Type', 'Last Modified')
        self.assertEqual(collist, columns)
        datalist = ((object_fakes.object_name_1, object_fakes.object_bytes_1, object_fakes.object_hash_1, object_fakes.object_content_type_1, object_fakes.object_modified_1), (object_fakes.object_name_2, object_fakes.object_bytes_2, object_fakes.object_hash_2, object_fakes.object_content_type_2, object_fakes.object_modified_2))
        self.assertEqual(datalist, tuple(data))

    def test_object_list_objects_all(self, o_mock):
        o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT), copy.deepcopy(object_fakes.OBJECT_2)]
        arglist = ['--all', object_fakes.container_name]
        verifylist = [('all', True), ('container', object_fakes.container_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'full_listing': True}
        o_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
        self.assertEqual(self.columns, columns)
        datalist = ((object_fakes.object_name_1,), (object_fakes.object_name_2,))
        self.assertEqual(datalist, tuple(data))