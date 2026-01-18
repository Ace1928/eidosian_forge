from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
class DriverManagerTest(testtools.TestCase):

    def setUp(self):
        super(DriverManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = driver.DriverManager(self.api)

    def test_driver_list(self):
        drivers = self.mgr.list()
        expect = [('GET', '/v1/drivers', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(drivers, matchers.HasLength(1))

    def test_driver_show(self):
        driver_ = self.mgr.get(DRIVER1['name'])
        expect = [('GET', '/v1/drivers/%s' % DRIVER1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        driver_attr = {}
        for attr in DRIVER1.keys():
            driver_attr[attr] = getattr(driver_, attr)
        self.assertEqual(DRIVER1, driver_attr)

    def test_driver_list_fields(self):
        drivers = self.mgr.list(fields=['name', 'hosts'])
        expect = [('GET', '/v1/drivers/?fields=name,hosts', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(drivers, matchers.HasLength(1))

    def test_driver_show_fields(self):
        driver_ = self.mgr.get(DRIVER1['name'], fields=['name', 'hosts'])
        expect = [('GET', '/v1/drivers/%s?fields=name,hosts' % DRIVER1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(DRIVER1['name'], driver_.name)
        self.assertEqual(DRIVER1['hosts'], driver_.hosts)

    def test_driver_properties(self):
        properties = self.mgr.properties(DRIVER2['name'])
        expect = [('GET', '/v1/drivers/%s/properties' % DRIVER2['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(DRIVER2_PROPERTIES, properties)

    def test_driver_raid_logical_disk_properties(self):
        properties = self.mgr.raid_logical_disk_properties(DRIVER2['name'])
        expect = [('GET', '/v1/drivers/%s/raid/logical_disk_properties' % DRIVER2['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(DRIVER2_RAID_LOGICAL_DISK_PROPERTIES, properties)

    @mock.patch.object(driver.DriverManager, '_list', autospec=True)
    def test_driver_raid_logical_disk_properties_indexerror(self, _list_mock):
        _list_mock.side_effect = IndexError
        properties = self.mgr.raid_logical_disk_properties(DRIVER2['name'])
        _list_mock.assert_called_once_with(self.mgr, '/v1/drivers/%s/raid/logical_disk_properties' % DRIVER2['name'], os_ironic_api_version=None, global_request_id=None)
        self.assertEqual({}, properties)

    @mock.patch.object(driver.DriverManager, 'update', autospec=True)
    def test_vendor_passthru_update(self, update_mock):
        vendor_passthru_args = {'arg1': 'val1'}
        kwargs = {'driver_name': 'driver_name', 'method': 'method', 'args': vendor_passthru_args}
        final_path = 'driver_name/vendor_passthru/method'
        for http_method in ('POST', 'PUT', 'PATCH'):
            kwargs['http_method'] = http_method
            self.mgr.vendor_passthru(**kwargs)
            update_mock.assert_called_once_with(mock.ANY, final_path, vendor_passthru_args, http_method=http_method, os_ironic_api_version=None, global_request_id=None)
            update_mock.reset_mock()

    @mock.patch.object(driver.DriverManager, 'get', autospec=True)
    def test_vendor_passthru_get(self, get_mock):
        kwargs = {'driver_name': 'driver_name', 'method': 'method', 'http_method': 'GET'}
        final_path = 'driver_name/vendor_passthru/method'
        self.mgr.vendor_passthru(**kwargs)
        get_mock.assert_called_once_with(mock.ANY, final_path, os_ironic_api_version=None, global_request_id=None)

    @mock.patch.object(driver.DriverManager, 'delete', autospec=True)
    def test_vendor_passthru_delete(self, delete_mock):
        kwargs = {'driver_name': 'driver_name', 'method': 'method', 'http_method': 'DELETE'}
        final_path = 'driver_name/vendor_passthru/method'
        self.mgr.vendor_passthru(**kwargs)
        delete_mock.assert_called_once_with(mock.ANY, final_path, os_ironic_api_version=None, global_request_id=None)

    @mock.patch.object(driver.DriverManager, 'delete', autospec=True)
    def test_vendor_passthru_unknown_http_method(self, delete_mock):
        kwargs = {'driver_name': 'driver_name', 'method': 'method', 'http_method': 'UNKNOWN'}
        self.assertRaises(exc.InvalidAttribute, self.mgr.vendor_passthru, **kwargs)

    def test_vendor_passthru_methods(self):
        vendor_methods = self.mgr.get_vendor_passthru_methods(DRIVER1['name'])
        expect = [('GET', '/v1/drivers/%s/vendor_passthru/methods' % DRIVER1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(DRIVER_VENDOR_PASSTHRU_METHOD, vendor_methods)