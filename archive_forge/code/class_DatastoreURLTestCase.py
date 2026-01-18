from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class DatastoreURLTestCase(base.TestCase):
    """Test the DatastoreURL object."""

    def test_path_strip(self):
        scheme = 'https'
        server = '13.37.73.31'
        path = 'images/ubuntu-14.04.vmdk'
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = datastore.DatastoreURL(scheme, server, path, dc_path, ds_name)
        expected_url = '%s://%s/folder/%s?%s' % (scheme, server, path, query)
        self.assertEqual(expected_url, str(url))

    def test_path_lstrip(self):
        scheme = 'https'
        server = '13.37.73.31'
        path = '/images/ubuntu-14.04.vmdk'
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = datastore.DatastoreURL(scheme, server, path, dc_path, ds_name)
        expected_url = '%s://%s/folder/%s?%s' % (scheme, server, path.lstrip('/'), query)
        self.assertEqual(expected_url, str(url))

    def test_path_rstrip(self):
        scheme = 'https'
        server = '13.37.73.31'
        path = 'images/ubuntu-14.04.vmdk/'
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = datastore.DatastoreURL(scheme, server, path, dc_path, ds_name)
        expected_url = '%s://%s/folder/%s?%s' % (scheme, server, path.rstrip('/'), query)
        self.assertEqual(expected_url, str(url))

    def test_urlparse(self):
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
        ds_url = datastore.DatastoreURL.urlparse(url)
        self.assertEqual(url, str(ds_url))

    def test_datastore_name(self):
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
        ds_url = datastore.DatastoreURL.urlparse(url)
        self.assertEqual(ds_name, ds_url.datastore_name)

    def test_datacenter_path(self):
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
        ds_url = datastore.DatastoreURL.urlparse(url)
        self.assertEqual(dc_path, ds_url.datacenter_path)

    def test_path(self):
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        path = 'images/aa.vmdk'
        query = urlparse.urlencode(params)
        url = 'https://13.37.73.31/folder/%s?%s' % (path, query)
        ds_url = datastore.DatastoreURL.urlparse(url)
        self.assertEqual(path, ds_url.path)

    @mock.patch('http.client.HTTPSConnection')
    def test_connect(self, mock_conn):
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
        ds_url = datastore.DatastoreURL.urlparse(url)
        cookie = mock.Mock()
        ds_url.connect('PUT', 128, cookie)
        mock_conn.assert_called_once_with('13.37.73.31')

    def test_get_transfer_ticket(self):
        dc_path = 'datacenter-1'
        ds_name = 'datastore-1'
        params = {'dcPath': dc_path, 'dsName': ds_name}
        query = urlparse.urlencode(params)
        url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
        session = mock.Mock()
        session.invoke_api = mock.Mock()

        class Ticket(object):
            id = 'fake_id'
        session.invoke_api.return_value = Ticket()
        ds_url = datastore.DatastoreURL.urlparse(url)
        ticket = ds_url.get_transfer_ticket(session, 'PUT')
        self.assertEqual('%s="%s"' % (constants.CGI_COOKIE_KEY, 'fake_id'), ticket)

    def test_get_datastore_by_ref(self):
        session = mock.Mock()
        ds_ref = mock.Mock()
        expected_props = {'summary.name': 'datastore1', 'summary.type': 'NFS', 'summary.freeSpace': 1000, 'summary.capacity': 2000}
        session.invoke_api = mock.Mock()
        session.invoke_api.return_value = expected_props
        ds_obj = datastore.get_datastore_by_ref(session, ds_ref)
        self.assertEqual(expected_props['summary.name'], ds_obj.name)
        self.assertEqual(expected_props['summary.type'], ds_obj.type)
        self.assertEqual(expected_props['summary.freeSpace'], ds_obj.freespace)
        self.assertEqual(expected_props['summary.capacity'], ds_obj.capacity)