import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
class VZStorageRemoteFSClientTestVase(RemoteFsClientTestCase):

    @mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
    def test_vzstorage_by_cluster_name(self, mock_read_mounts):
        client = remotefs.VZStorageRemoteFSClient('vzstorage', root_helper='true', vzstorage_mount_point_base='/mnt')
        share = 'qwe'
        cluster_name = share
        mount_point = client.get_mount_point(share)
        client.mount(share)
        calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('pstorage-mount', '-c', cluster_name, mount_point, root_helper='true', check_exit_code=0, run_as_root=True)]
        self.mock_execute.assert_has_calls(calls)

    @mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
    def test_vzstorage_with_auth(self, mock_read_mounts):
        client = remotefs.VZStorageRemoteFSClient('vzstorage', root_helper='true', vzstorage_mount_point_base='/mnt')
        cluster_name = 'qwe'
        password = '123456'
        share = '%s:%s' % (cluster_name, password)
        mount_point = client.get_mount_point(share)
        client.mount(share)
        calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('pstorage', '-c', cluster_name, 'auth-node', '-P', process_input=password, root_helper='true', run_as_root=True), mock.call('pstorage-mount', '-c', cluster_name, mount_point, root_helper='true', check_exit_code=0, run_as_root=True)]
        self.mock_execute.assert_has_calls(calls)

    @mock.patch('os.path.exists', return_value=False)
    @mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
    def test_vzstorage_with_mds_list(self, mock_read_mounts, mock_exists):
        client = remotefs.VZStorageRemoteFSClient('vzstorage', root_helper='true', vzstorage_mount_point_base='/mnt')
        cluster_name = 'qwe'
        mds_list = ['10.0.0.1', '10.0.0.2']
        share = '%s:/%s' % (','.join(mds_list), cluster_name)
        mount_point = client.get_mount_point(share)
        vz_conf_dir = os.path.join('/etc/pstorage/clusters/', cluster_name)
        tmp_dir = '/tmp/fake_dir/'
        with mock.patch.object(tempfile, 'mkdtemp', return_value=tmp_dir):
            with mock.patch('os_brick.remotefs.remotefs.open', new_callable=mock.mock_open) as mock_open:
                client.mount(share)
                write_calls = [mock.call(tmp_dir + 'bs_list', 'w'), mock.call().__enter__(), mock.call().write('10.0.0.1\n'), mock.call().write('10.0.0.2\n'), mock.call().__exit__(None, None, None)]
                mock_open.assert_has_calls(write_calls)
        calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('cp', '-rf', tmp_dir, vz_conf_dir, run_as_root=True, root_helper='true'), mock.call('chown', '-R', 'root:root', vz_conf_dir, run_as_root=True, root_helper='true'), mock.call('pstorage-mount', '-c', cluster_name, mount_point, root_helper='true', check_exit_code=0, run_as_root=True)]
        self.mock_execute.assert_has_calls(calls)

    @mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
    def test_vzstorage_invalid_share(self, mock_read_mounts):
        client = remotefs.VZStorageRemoteFSClient('vzstorage', root_helper='true', vzstorage_mount_point_base='/mnt')
        self.assertRaises(exception.BrickException, client.mount, ':')