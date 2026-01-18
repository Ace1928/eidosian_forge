import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class RequestsTransportTest(base.TestCase):
    """Tests for RequestsTransport."""

    def test_open(self):
        transport = service.RequestsTransport()
        data = b'Hello World'
        resp = mock.Mock(content=data)
        transport.session.get = mock.Mock(return_value=resp)
        request = mock.Mock(url=mock.sentinel.url)
        self.assertEqual(data, transport.open(request).getvalue())
        transport.session.get.assert_called_once_with(mock.sentinel.url, verify=transport.verify)

    def test_send(self):
        transport = service.RequestsTransport()
        resp = mock.Mock(status_code=mock.sentinel.status_code, headers=mock.sentinel.headers, content=mock.sentinel.content)
        transport.session.post = mock.Mock(return_value=resp)
        request = mock.Mock(url=mock.sentinel.url, message=mock.sentinel.message, headers=mock.sentinel.req_headers)
        reply = transport.send(request)
        self.assertEqual(mock.sentinel.status_code, reply.code)
        self.assertEqual(mock.sentinel.headers, reply.headers)
        self.assertEqual(mock.sentinel.content, reply.message)

    def test_set_conn_pool_size(self):
        transport = service.RequestsTransport(pool_maxsize=100)
        local_file_adapter = transport.session.adapters['file:///']
        self.assertEqual(100, local_file_adapter._pool_connections)
        self.assertEqual(100, local_file_adapter._pool_maxsize)
        https_adapter = transport.session.adapters['https://']
        self.assertEqual(100, https_adapter._pool_connections)
        self.assertEqual(100, https_adapter._pool_maxsize)

    @mock.patch('os.path.getsize')
    def test_send_with_local_file_url(self, get_size_mock):
        transport = service.RequestsTransport()
        url = 'file:///foo'
        request = requests.Request('GET', url).prepare()
        data = b'Hello World'
        get_size_mock.return_value = len(data)

        def read_mock():
            return data
        open_mock = mock.MagicMock(name='file_handle', spec=open)
        file_spec = list(set(dir(io.TextIOWrapper)).union(set(dir(io.BytesIO))))
        file_handle = mock.MagicMock(spec=file_spec)
        file_handle.write.return_value = None
        file_handle.__enter__.return_value = file_handle
        file_handle.read.side_effect = read_mock
        open_mock.return_value = file_handle
        with mock.patch('builtins.open', open_mock, create=True):
            resp = transport.session.send(request)
            self.assertEqual(data, resp.content)

    def test_send_with_connection_timeout(self):
        transport = service.RequestsTransport(connection_timeout=120)
        request = mock.Mock(url=mock.sentinel.url, message=mock.sentinel.message, headers=mock.sentinel.req_headers)
        with mock.patch.object(transport.session, 'post') as mock_post:
            transport.send(request)
            mock_post.assert_called_once_with(mock.sentinel.url, data=mock.sentinel.message, headers=mock.sentinel.req_headers, timeout=120, verify=transport.verify)