from unittest import mock
from openstack.compute.v2 import flavor as _flavor
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import flavor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestFlavorUnset(TestFlavor):
    flavor = compute_fakes.create_one_flavor(attrs={'os-flavor-access:is_public': False})
    project = identity_fakes.FakeProject.create_one_project()

    def setUp(self):
        super(TestFlavorUnset, self).setUp()
        self.compute_sdk_client.find_flavor.return_value = self.flavor
        self.projects_mock.get.return_value = self.project
        self.cmd = flavor.UnsetFlavor(self.app, None)
        self.mock_shortcut = self.compute_sdk_client.delete_flavor_extra_specs_property

    def test_flavor_unset_property(self):
        arglist = ['--property', 'property', 'baremetal']
        verifylist = [('properties', ['property']), ('flavor', 'baremetal')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        self.mock_shortcut.assert_called_with(self.flavor.id, 'property')
        self.compute_sdk_client.flavor_remove_tenant_access.assert_not_called()
        self.assertIsNone(result)

    def test_flavor_unset_properties(self):
        arglist = ['--property', 'property1', '--property', 'property2', 'baremetal']
        verifylist = [('properties', ['property1', 'property2']), ('flavor', 'baremetal')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        calls = [mock.call(self.flavor.id, 'property1'), mock.call(self.flavor.id, 'property2')]
        self.mock_shortcut.assert_has_calls(calls)
        calls.append(mock.call(self.flavor.id, 'property'))
        self.assertRaises(AssertionError, self.mock_shortcut.assert_has_calls, calls)
        self.compute_sdk_client.flavor_remove_tenant_access.assert_not_called()

    def test_flavor_unset_project(self):
        arglist = ['--project', self.project.id, self.flavor.id]
        verifylist = [('project', self.project.id), ('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        self.compute_sdk_client.flavor_remove_tenant_access.assert_called_with(self.flavor.id, self.project.id)
        self.compute_sdk_client.delete_flavor_extra_specs_property.assert_not_called()
        self.assertIsNone(result)

    def test_flavor_unset_no_project(self):
        arglist = ['--project', self.flavor.id]
        verifylist = [('project', None), ('flavor', self.flavor.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_flavor_unset_no_flavor(self):
        arglist = ['--project', self.project.id]
        verifylist = [('project', self.project.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_flavor_unset_with_unexist_flavor(self):
        self.compute_sdk_client.find_flavor.side_effect = [sdk_exceptions.ResourceNotFound]
        arglist = ['--project', self.project.id, 'unexist_flavor']
        verifylist = [('project', self.project.id), ('flavor', 'unexist_flavor')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_flavor_unset_nothing(self):
        arglist = [self.flavor.id]
        verifylist = [('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.flavor_remove_tenant_access.assert_not_called()