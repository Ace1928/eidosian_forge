import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _is_crypt_device_available(self, dev_name):
    if not os.path.exists('/dev/mapper/%s' % dev_name):
        return False
    try:
        self._execute('cryptsetup', 'status', dev_name, run_as_root=True)
    except processutils.ProcessExecutionError as e:
        if e.exit_code != 1:
            LOG.warning('cryptsetup status %(dev_name)s exited abnormally (status %(exit_code)s): %(err)s', {'dev_name': dev_name, 'exit_code': e.exit_code, 'err': e.stderr})
        return False
    return True