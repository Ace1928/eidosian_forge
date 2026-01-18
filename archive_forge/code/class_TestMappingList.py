import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestMappingList(TestMapping):

    def setUp(self):
        super(TestMappingList, self).setUp()
        self.mapping_mock.get.return_value = fakes.FakeResource(None, {'id': identity_fakes.mapping_id}, loaded=True)
        self.mapping_mock.list.return_value = [fakes.FakeResource(None, {'id': identity_fakes.mapping_id, 'schema_version': '1.0'}, loaded=True), fakes.FakeResource(None, {'id': 'extra_mapping', 'schema_version': '2.0'}, loaded=True)]
        self.cmd = mapping.ListMapping(self.app, None)

    def test_mapping_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.mapping_mock.list.assert_called_with()
        collist = ('ID', 'schema_version')
        self.assertEqual(collist, columns)
        datalist = [(identity_fakes.mapping_id, '1.0'), ('extra_mapping', '2.0')]
        self.assertEqual(datalist, data)