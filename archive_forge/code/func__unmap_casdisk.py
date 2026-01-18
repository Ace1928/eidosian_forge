import os
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import caches
from os_brick import exception
from os_brick import executor
def _unmap_casdisk(self, coreid):
    cmd = ['casadm', '-R', '-f', '-i', self.cache_id, '-j', coreid]
    kwargs = dict(run_as_root=True, root_helper=self._root_helper)
    out, err = self.os_execute(*cmd, **kwargs)