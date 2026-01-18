from __future__ import (absolute_import, division, print_function)
import os
import re
import time
import glob
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible_collections.ansible.netcommon.plugins.action.network import ActionModule as ActionNetworkModule
from ansible.utils.display import Display
def _write_backup(self, host, contents):
    backup_path = self._get_working_path() + '/backup'
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)
    for fn in glob.glob('%s/%s*' % (backup_path, host)):
        os.remove(fn)
    tstamp = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime(time.time()))
    filename = '%s/%s_config.%s' % (backup_path, host, tstamp)
    fh = open(filename, 'w')
    fh.write(contents)
    fh.close()
    return filename