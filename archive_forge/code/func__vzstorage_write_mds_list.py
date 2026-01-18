import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
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