from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
def _force(value):
    if value == '':
        value = None
    return cmd_runner_fmt.as_optval('--force-')(value, ctx_ignore_none=True)