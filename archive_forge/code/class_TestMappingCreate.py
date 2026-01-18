import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestMappingCreate(TestMapping):

    def setUp(self):
        super(TestMappingCreate, self).setUp()
        self.mapping_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.MAPPING_RESPONSE), loaded=True)
        self.cmd = mapping.CreateMapping(self.app, None)

    def test_create_mapping(self):
        arglist = ['--rules', identity_fakes.mapping_rules_file_path, identity_fakes.mapping_id]
        verifylist = [('mapping', identity_fakes.mapping_id), ('rules', identity_fakes.mapping_rules_file_path)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = identity_fakes.MAPPING_RULES
        with mock.patch('openstackclient.identity.v3.mapping.CreateMapping._read_rules', mocker):
            columns, data = self.cmd.take_action(parsed_args)
        self.mapping_mock.create.assert_called_with(mapping_id=identity_fakes.mapping_id, rules=identity_fakes.MAPPING_RULES, schema_version=None)
        collist = ('id', 'rules')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.mapping_id, identity_fakes.MAPPING_RULES)
        self.assertEqual(datalist, data)