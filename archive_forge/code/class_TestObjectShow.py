import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import object as obj
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
@mock.patch('openstackclient.api.object_store_v1.APIv1.object_show')
class TestObjectShow(TestObject):

    def setUp(self):
        super(TestObjectShow, self).setUp()
        self.cmd = obj.ShowObject(self.app, None)

    def test_object_show(self, c_mock):
        c_mock.return_value = copy.deepcopy(object_fakes.OBJECT)
        arglist = [object_fakes.container_name, object_fakes.object_name_1]
        verifylist = [('container', object_fakes.container_name), ('object', object_fakes.object_name_1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {}
        c_mock.assert_called_with(container=object_fakes.container_name, object=object_fakes.object_name_1, **kwargs)
        collist = ('bytes', 'content_type', 'hash', 'last_modified', 'name')
        self.assertEqual(collist, columns)
        datalist = (object_fakes.object_bytes_1, object_fakes.object_content_type_1, object_fakes.object_hash_1, object_fakes.object_modified_1, object_fakes.object_name_1)
        self.assertEqual(datalist, data)