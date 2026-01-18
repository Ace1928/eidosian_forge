from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _generic_state_action(self, actionable_func, actionable_var, params):
    actionable_snaps = [s for s in self.vars.name if actionable_func(s)]
    if not actionable_snaps:
        return
    self.changed = True
    self.vars[actionable_var] = actionable_snaps
    if self.check_mode:
        return
    self.vars.cmd, rc, out, err, run_info = self._run_multiple_commands(params, actionable_snaps)
    self.vars.run_info = run_info
    if rc == 0:
        return
    msg = "Ooops! Snap operation failed while executing '{cmd}', please examine logs and error output for more details.".format(cmd=self.vars.cmd)
    self.do_raise(msg=msg)