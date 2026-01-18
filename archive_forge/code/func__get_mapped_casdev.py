import os
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import caches
from os_brick import exception
from os_brick import executor
def _get_mapped_casdev(self, core):
    cmd = ['casadm', '-L']
    kwargs = dict(run_as_root=True, root_helper=self._root_helper)
    out, err = self.os_execute(*cmd, **kwargs)
    for line in out.splitlines():
        if line.find(core) < 0:
            continue
        fields = line.split()
        return fields[5]
    raise exception.BrickException('Cannot find emulated device.')