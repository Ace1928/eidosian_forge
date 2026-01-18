from unittest import mock
from openstackclient.identity.v2_0 import catalog
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCatalogShow(TestCatalog):

    def setUp(self):
        super(TestCatalogShow, self).setUp()
        self.cmd = catalog.ShowCatalog(self.app, None)

    def test_catalog_show(self):
        auth_ref = identity_fakes.fake_auth_ref(identity_fakes.UNSCOPED_TOKEN, fake_service=self.service_catalog)
        self.ar_mock = mock.PropertyMock(return_value=auth_ref)
        type(self.app.client_manager).auth_ref = self.ar_mock
        arglist = ['compute']
        verifylist = [('service', 'compute')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('endpoints', 'id', 'name', 'type')
        self.assertEqual(collist, columns)
        datalist = (catalog.EndpointsColumn(auth_ref.service_catalog.catalog[0]['endpoints']), self.service_catalog.id, 'supernova', 'compute')
        self.assertCountEqual(datalist, data)