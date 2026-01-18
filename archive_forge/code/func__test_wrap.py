import os
import ssl
from unittest import mock
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
@mock.patch('ssl.wrap_socket')
@mock.patch('os.path.exists')
def _test_wrap(self, exists_mock, wrap_socket_mock, **kwargs):
    exists_mock.return_value = True
    sock = mock.Mock()
    self.conf.set_default('cert_file', self.cert_file_name, group=sslutils.config_section)
    self.conf.set_default('key_file', self.key_file_name, group=sslutils.config_section)
    ssl_kwargs = {'server_side': True, 'certfile': self.conf.ssl.cert_file, 'keyfile': self.conf.ssl.key_file, 'cert_reqs': ssl.CERT_NONE}
    if kwargs:
        ssl_kwargs.update(**kwargs)
    sslutils.wrap(self.conf, sock)
    wrap_socket_mock.assert_called_once_with(sock, **ssl_kwargs)