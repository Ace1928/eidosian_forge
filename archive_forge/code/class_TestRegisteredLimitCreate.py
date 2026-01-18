import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegisteredLimitCreate(TestRegisteredLimit):

    def setUp(self):
        super(TestRegisteredLimitCreate, self).setUp()
        self.service = fakes.FakeResource(None, copy.deepcopy(identity_fakes.SERVICE), loaded=True)
        self.services_mock.get.return_value = self.service
        self.region = fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGION), loaded=True)
        self.regions_mock.get.return_value = self.region
        self.cmd = registered_limit.CreateRegisteredLimit(self.app, None)

    def test_registered_limit_create_without_options(self):
        self.registered_limit_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGISTERED_LIMIT), loaded=True)
        resource_name = identity_fakes.registered_limit_resource_name
        default_limit = identity_fakes.registered_limit_default_limit
        arglist = ['--service', identity_fakes.service_id, '--default-limit', '10', resource_name]
        verifylist = [('service', identity_fakes.service_id), ('default_limit', default_limit), ('resource_name', resource_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'region': None}
        self.registered_limit_mock.create.assert_called_with(self.service, resource_name, default_limit, **kwargs)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, None, identity_fakes.registered_limit_id, None, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)

    def test_registered_limit_create_with_options(self):
        self.registered_limit_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGISTERED_LIMIT_OPTIONS), loaded=True)
        resource_name = identity_fakes.registered_limit_resource_name
        default_limit = identity_fakes.registered_limit_default_limit
        description = identity_fakes.registered_limit_description
        arglist = ['--region', identity_fakes.region_id, '--description', description, '--service', identity_fakes.service_id, '--default-limit', '10', resource_name]
        verifylist = [('region', identity_fakes.region_id), ('description', description), ('service', identity_fakes.service_id), ('default_limit', default_limit), ('resource_name', resource_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': description, 'region': self.region}
        self.registered_limit_mock.create.assert_called_with(self.service, resource_name, default_limit, **kwargs)
        collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.registered_limit_default_limit, description, identity_fakes.registered_limit_id, identity_fakes.region_id, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)