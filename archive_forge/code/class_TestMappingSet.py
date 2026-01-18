import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestMappingSet(TestMapping):

    def setUp(self):
        super(TestMappingSet, self).setUp()
        self.mapping_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.MAPPING_RESPONSE), loaded=True)
        self.mapping_mock.update.return_value = fakes.FakeResource(None, identity_fakes.MAPPING_RESPONSE_2, loaded=True)
        self.cmd = mapping.SetMapping(self.app, None)

    def test_set_new_rules(self):
        arglist = ['--rules', identity_fakes.mapping_rules_file_path, identity_fakes.mapping_id]
        verifylist = [('mapping', identity_fakes.mapping_id), ('rules', identity_fakes.mapping_rules_file_path)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = identity_fakes.MAPPING_RULES_2
        with mock.patch('openstackclient.identity.v3.mapping.SetMapping._read_rules', mocker):
            result = self.cmd.take_action(parsed_args)
        self.mapping_mock.update.assert_called_with(mapping=identity_fakes.mapping_id, rules=identity_fakes.MAPPING_RULES_2, schema_version=None)
        self.assertIsNone(result)

    def test_set_rules_wrong_file_path(self):
        arglist = ['--rules', identity_fakes.mapping_rules_file_path, identity_fakes.mapping_id]
        verifylist = [('mapping', identity_fakes.mapping_id), ('rules', identity_fakes.mapping_rules_file_path)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)