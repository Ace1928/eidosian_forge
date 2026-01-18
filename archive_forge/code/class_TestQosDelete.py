import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
class TestQosDelete(TestQos):
    qos_specs = volume_fakes.create_qoses(count=2)

    def setUp(self):
        super(TestQosDelete, self).setUp()
        self.qos_mock.get = volume_fakes.get_qoses(self.qos_specs)
        self.cmd = qos_specs.DeleteQos(self.app, None)

    def test_qos_delete(self):
        arglist = [self.qos_specs[0].id]
        verifylist = [('qos_specs', [self.qos_specs[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.qos_mock.delete.assert_called_with(self.qos_specs[0].id, False)
        self.assertIsNone(result)

    def test_qos_delete_with_force(self):
        arglist = ['--force', self.qos_specs[0].id]
        verifylist = [('force', True), ('qos_specs', [self.qos_specs[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.qos_mock.delete.assert_called_with(self.qos_specs[0].id, True)
        self.assertIsNone(result)

    def test_delete_multiple_qoses(self):
        arglist = []
        for q in self.qos_specs:
            arglist.append(q.id)
        verifylist = [('qos_specs', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for q in self.qos_specs:
            calls.append(call(q.id, False))
        self.qos_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_qoses_with_exception(self):
        arglist = [self.qos_specs[0].id, 'unexist_qos']
        verifylist = [('qos_specs', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self.qos_specs[0], exceptions.CommandError]
        with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
            try:
                self.cmd.take_action(parsed_args)
                self.fail('CommandError should be raised.')
            except exceptions.CommandError as e:
                self.assertEqual('1 of 2 QoS specifications failed to delete.', str(e))
            find_mock.assert_any_call(self.qos_mock, self.qos_specs[0].id)
            find_mock.assert_any_call(self.qos_mock, 'unexist_qos')
            self.assertEqual(2, find_mock.call_count)
            self.qos_mock.delete.assert_called_once_with(self.qos_specs[0].id, False)