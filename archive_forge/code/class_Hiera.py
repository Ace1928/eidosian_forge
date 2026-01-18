from __future__ import (absolute_import, division, print_function)
from ansible.plugins.lookup import LookupBase
from ansible.utils.cmd_functions import run_cmd
from ansible.module_utils.common.text.converters import to_text
class Hiera(object):

    def __init__(self, hiera_cfg, hiera_bin):
        self.hiera_cfg = hiera_cfg
        self.hiera_bin = hiera_bin

    def get(self, hiera_key):
        pargs = [self.hiera_bin]
        pargs.extend(['-c', self.hiera_cfg])
        pargs.extend(hiera_key)
        rc, output, err = run_cmd('{0} -c {1} {2}'.format(self.hiera_bin, self.hiera_cfg, hiera_key[0]))
        return to_text(output.strip())