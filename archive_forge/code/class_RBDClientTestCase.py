from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
class RBDClientTestCase(base.TestCase):

    def setUp(self):
        super(RBDClientTestCase, self).setUp()

    @mock.patch('os_brick.initiator.linuxrbd.rbd')
    @mock.patch('os_brick.initiator.linuxrbd.rados')
    def test_with_client(self, mock_rados, mock_rbd):
        with linuxrbd.RBDClient('test_user', 'test_pool') as client:
            self.assertEqual('/etc/ceph/ceph.conf', client.rbd_conf)
            self.assertEqual(utils.convert_str('test_user'), client.rbd_user)
            self.assertEqual(utils.convert_str('test_pool'), client.rbd_pool)
            mock_rados.Rados.assert_called_once_with(clustername='ceph', rados_id=utils.convert_str('test_user'), conffile='/etc/ceph/ceph.conf')
            self.assertEqual(1, mock_rados.Rados.return_value.connect.call_count)
            mock_rados.Rados.return_value.open_ioctx.assert_called_once_with(utils.convert_str('test_pool'))
        self.assertEqual(1, mock_rados.Rados.return_value.shutdown.call_count)

    @mock.patch.object(MockRados.Rados, 'connect', side_effect=MockRados.Error)
    def test_with_client_error(self, _):
        linuxrbd.rados = MockRados
        linuxrbd.rados.Error = MockRados.Error

        def test():
            with linuxrbd.RBDClient('test_user', 'test_pool'):
                pass
        self.assertRaises(exception.BrickException, test)