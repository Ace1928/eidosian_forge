from unittest import mock
from openstackclient.identity.v2_0 import catalog
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCatalogList(TestCatalog):
    columns = ('Name', 'Type', 'Endpoints')

    def setUp(self):
        super(TestCatalogList, self).setUp()
        self.cmd = catalog.ListCatalog(self.app, None)

    def test_catalog_list(self):
        auth_ref = identity_fakes.fake_auth_ref(identity_fakes.TOKEN, fake_service=self.service_catalog)
        self.ar_mock = mock.PropertyMock(return_value=auth_ref)
        type(self.app.client_manager).auth_ref = self.ar_mock
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = (('supernova', 'compute', catalog.EndpointsColumn(auth_ref.service_catalog.catalog[0]['endpoints'])),)
        self.assertCountEqual(datalist, tuple(data))

    def test_catalog_list_with_endpoint_url(self):
        attr = {'id': 'qwertyuiop', 'type': 'compute', 'name': 'supernova', 'endpoints': [{'region': 'one', 'publicURL': 'https://public.one.example.com'}, {'region': 'two', 'publicURL': 'https://public.two.example.com', 'internalURL': 'https://internal.two.example.com'}]}
        service_catalog = identity_fakes.FakeCatalog.create_catalog(attr)
        auth_ref = identity_fakes.fake_auth_ref(identity_fakes.TOKEN, fake_service=service_catalog)
        self.ar_mock = mock.PropertyMock(return_value=auth_ref)
        type(self.app.client_manager).auth_ref = self.ar_mock
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = (('supernova', 'compute', catalog.EndpointsColumn(auth_ref.service_catalog.catalog[0]['endpoints'])),)
        self.assertCountEqual(datalist, tuple(data))