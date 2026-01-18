import copy
import io
from unittest import mock
from osc_lib import exceptions
from requests_mock.contrib import fixture
from openstackclient.object.v1 import object as object_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
class TestObjectCreate(TestObjectAll):

    def setUp(self):
        super(TestObjectCreate, self).setUp()
        self.cmd = object_cmds.CreateObject(self.app, None)

    def test_multiple_object_create_with_object_name(self):
        arglist = [object_fakes.container_name, object_fakes.object_name_1, object_fakes.object_name_2, '--name', object_fakes.object_upload_name]
        verifylist = [('container', object_fakes.container_name), ('objects', [object_fakes.object_name_1, object_fakes.object_name_2]), ('name', object_fakes.object_upload_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)