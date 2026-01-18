import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
class VZStorageRemoteFSClient(RemoteFsClient):

    def _vzstorage_write_mds_list(self, cluster_name, mdss):
        tmp_dir = tempfile.mkdtemp(prefix='vzstorage-')
        tmp_bs_path = os.path.join(tmp_dir, 'bs_list')
        with open(tmp_bs_path, 'w') as f:
            for mds in mdss:
                f.write(mds + '\n')
        conf_dir = os.path.join('/etc/pstorage/clusters', cluster_name)
        if os.path.exists(conf_dir):
            bs_path = os.path.join(conf_dir, 'bs_list')
            self._execute('cp', '-f', tmp_bs_path, bs_path, root_helper=self._root_helper, run_as_root=True)
        else:
            self._execute('cp', '-rf', tmp_dir, conf_dir, root_helper=self._root_helper, run_as_root=True)
        self._execute('chown', '-R', 'root:root', conf_dir, root_helper=self._root_helper, run_as_root=True)

    def _do_mount(self, mount_type, vz_share, mount_path, mount_options=None, flags=None):
        m = re.search('(?:(\\S+):\\/)?([a-zA-Z0-9_-]+)(?::(\\S+))?', vz_share)
        if not m:
            msg = _('Invalid Virtuozzo Storage share specification: %r.Must be: [MDS1[,MDS2],...:/]<CLUSTER NAME>[:PASSWORD].') % vz_share
            raise exception.BrickException(msg)
        mdss = m.group(1)
        cluster_name = m.group(2)
        passwd = m.group(3)
        if mdss:
            mdss = mdss.split(',')
            self._vzstorage_write_mds_list(cluster_name, mdss)
        if passwd:
            self._execute('pstorage', '-c', cluster_name, 'auth-node', '-P', process_input=passwd, root_helper=self._root_helper, run_as_root=True)
        mnt_cmd = ['pstorage-mount', '-c', cluster_name]
        if flags:
            mnt_cmd.extend(flags)
        mnt_cmd.extend([mount_path])
        self._execute(*mnt_cmd, root_helper=self._root_helper, run_as_root=True, check_exit_code=0)