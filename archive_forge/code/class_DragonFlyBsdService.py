from __future__ import absolute_import, division, print_function
import glob
import json
import os
import platform
import re
import select
import shlex
import subprocess
import tempfile
import time
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.six import PY2, b
class DragonFlyBsdService(FreeBsdService):
    """
    This is the DragonFly BSD Service manipulation class - it uses the /etc/rc.conf
    file for controlling services started at boot and the 'service' binary to
    check status and perform direct service manipulation.
    """
    platform = 'DragonFly'
    distribution = None

    def service_enable(self):
        if self.enable:
            self.rcconf_value = 'YES'
        else:
            self.rcconf_value = 'NO'
        rcfiles = ['/etc/rc.conf']
        for rcfile in rcfiles:
            if os.path.isfile(rcfile):
                self.rcconf_file = rcfile
        self.rcconf_key = '%s' % self.name.replace('-', '_')
        return self.service_enable_rcconf()