from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def _puppet_cmd(module):
    return module.get_bin_path('puppet', False, _PUPPET_PATH_PREFIX)