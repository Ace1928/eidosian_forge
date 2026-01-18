import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
class RemoteFsClientTestCase(base.TestCase):

    def setUp(self):
        super(RemoteFsClientTestCase, self).setUp()
        self.mock_execute = self.mock_object(priv_rootwrap, 'execute', return_value=None)

    @mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
    def test_cifs(self, mock_read_mounts):
        client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/mnt')
        share = '10.0.0.1:/qwe'
        mount_point = client.get_mount_point(share)
        client.mount(share)
        calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('mount', '-t', 'cifs', share, mount_point, run_as_root=True, root_helper='true', check_exit_code=0)]
        self.mock_execute.assert_has_calls(calls)

    @mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
    def test_nfs(self, mock_read_mounts):
        client = remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/mnt')
        share = '10.0.0.1:/qwe'
        mount_point = client.get_mount_point(share)
        client.mount(share)
        calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('mount', '-t', 'nfs', '-o', 'vers=4,minorversion=1', share, mount_point, check_exit_code=0, run_as_root=True, root_helper='true')]
        self.mock_execute.assert_has_calls(calls)

    def test_read_mounts(self):
        mounts = 'device1 mnt_point1 ext4 rw,seclabel,relatime 0 0\n                    device2 mnt_point2 ext4 rw,seclabel,relatime 0 0'
        with mock.patch('os_brick.remotefs.remotefs.open', mock.mock_open(read_data=mounts)) as mock_open:
            client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/mnt')
            ret = client._read_mounts()
            mock_open.assert_called_once_with('/proc/mounts', 'r')
        self.assertEqual(ret, {'mnt_point1': 'device1', 'mnt_point2': 'device2'})

    @mock.patch.object(priv_rootwrap, 'execute')
    @mock.patch.object(remotefs.RemoteFsClient, '_do_mount')
    def test_mount_already_mounted(self, mock_do_mount, mock_execute):
        share = '10.0.0.1:/share'
        client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/mnt')
        mounts = {client.get_mount_point(share): 'some_dev'}
        with mock.patch.object(client, '_read_mounts', return_value=mounts):
            client.mount(share)
            self.assertEqual(mock_do_mount.call_count, 0)
            self.assertEqual(mock_execute.call_count, 0)

    @mock.patch.object(priv_rootwrap, 'execute')
    def test_mount_race(self, mock_execute):
        err_msg = 'mount.nfs: /var/asdf is already mounted'
        mock_execute.side_effect = putils.ProcessExecutionError(stderr=err_msg)
        mounts = {'192.0.2.20:/share': '/var/asdf/'}
        client = remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/var/asdf')
        with mock.patch.object(client, '_read_mounts', return_value=mounts):
            client._do_mount('nfs', '192.0.2.20:/share', '/var/asdf')

    @mock.patch.object(priv_rootwrap, 'execute')
    def test_mount_failure(self, mock_execute):
        err_msg = 'mount.nfs: nfs broke'
        mock_execute.side_effect = putils.ProcessExecutionError(stderr=err_msg)
        client = remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/var/asdf')
        self.assertRaises(putils.ProcessExecutionError, client._do_mount, 'nfs', '192.0.2.20:/share', '/var/asdf')

    def _test_no_mount_point(self, fs_type):
        self.assertRaises(exception.InvalidParameterValue, remotefs.RemoteFsClient, fs_type, root_helper='true')

    def test_no_mount_point_nfs(self):
        self._test_no_mount_point('nfs')

    def test_no_mount_point_cifs(self):
        self._test_no_mount_point('cifs')

    def test_no_mount_point_glusterfs(self):
        self._test_no_mount_point('glusterfs')

    def test_no_mount_point_vzstorage(self):
        self._test_no_mount_point('vzstorage')

    def test_no_mount_point_quobyte(self):
        self._test_no_mount_point('quobyte')

    def test_invalid_fs(self):
        self.assertRaises(exception.ProtocolNotSupported, remotefs.RemoteFsClient, 'my_fs', root_helper='true')

    def test_init_sets_mount_base(self):
        client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/fake', cifs_mount_point_base='/fake2')
        self.assertEqual('/fake', client._mount_base)

    @mock.patch('os_brick.remotefs.remotefs.RemoteFsClient._check_nfs_options')
    def test_init_nfs_calls_check_nfs_options(self, mock_check_nfs_options):
        remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/fake')
        mock_check_nfs_options.assert_called_once_with()