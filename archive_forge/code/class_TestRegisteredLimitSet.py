import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegisteredLimitSet(TestRegisteredLimit):

    def setUp(self):
        super(TestRegisteredLimitSet, self).setUp()
        self.cmd = registered_limit.SetRegisteredLimit(self.app, None)

    def test_registered_limit_set_description(self):
        registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
        registered_limit['description'] = identity_fakes.registered_limit_description
        self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
        arglist = ['--description', identity_fakes.registered_limit_description, identity_fakes.registered_limit_id]
        verifylist = [('description', identity_fakes.registered_limit_description), ('registered_limit_id', identity_fakes.registered_limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=None, resource_name=None, default_limit=None, description=identity_fakes.registered_limit_description, region=None)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, identity_fakes.registered_limit_description, identity_fakes.registered_limit_id, None, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)

    def test_registered_limit_set_default_limit(self):
        registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
        default_limit = 20
        registered_limit['default_limit'] = default_limit
        self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
        arglist = ['--default-limit', str(default_limit), identity_fakes.registered_limit_id]
        verifylist = [('default_limit', default_limit), ('registered_limit_id', identity_fakes.registered_limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=None, resource_name=None, default_limit=default_limit, description=None, region=None)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (default_limit, None, identity_fakes.registered_limit_id, None, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)

    def test_registered_limit_set_resource_name(self):
        registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
        resource_name = 'volumes'
        registered_limit['resource_name'] = resource_name
        self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
        arglist = ['--resource-name', resource_name, identity_fakes.registered_limit_id]
        verifylist = [('resource_name', resource_name), ('registered_limit_id', identity_fakes.registered_limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=None, resource_name=resource_name, default_limit=None, description=None, region=None)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, None, identity_fakes.registered_limit_id, None, resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)

    def test_registered_limit_set_service(self):
        registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
        service = identity_fakes.FakeService.create_one_service()
        registered_limit['service_id'] = service.id
        self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
        self.services_mock.get.return_value = service
        arglist = ['--service', service.id, identity_fakes.registered_limit_id]
        verifylist = [('service', service.id), ('registered_limit_id', identity_fakes.registered_limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=service, resource_name=None, default_limit=None, description=None, region=None)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, None, identity_fakes.registered_limit_id, None, identity_fakes.registered_limit_resource_name, service.id)
        self.assertEqual(datalist, data)

    def test_registered_limit_set_region(self):
        registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
        region = identity_fakes.REGION
        region['id'] = 'RegionTwo'
        region = fakes.FakeResource(None, copy.deepcopy(region), loaded=True)
        registered_limit['region_id'] = region.id
        self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
        self.regions_mock.get.return_value = region
        arglist = ['--region', region.id, identity_fakes.registered_limit_id]
        verifylist = [('region', region.id), ('registered_limit_id', identity_fakes.registered_limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=None, resource_name=None, default_limit=None, description=None, region=region)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, None, identity_fakes.registered_limit_id, region.id, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)