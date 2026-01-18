from __future__ import (absolute_import, division, print_function)
import os
import os.path
import subprocess
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils._text import to_bytes
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
def _check_domain(self, domain):
    p = subprocess.Popen([self.virsh, '-q', '-c', 'lxc:///', 'dominfo', to_bytes(domain)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    if p.returncode:
        raise AnsibleError('%s is not a lxc defined in libvirt' % domain)