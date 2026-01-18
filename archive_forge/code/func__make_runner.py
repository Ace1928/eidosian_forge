from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper, ModuleHelperException
def _make_runner(self, lang):
    return CmdRunner(self.module, command=self.command, arg_formats=self.command_args_formats, force_lang=lang, check_rc=True)