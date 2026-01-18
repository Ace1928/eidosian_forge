from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
def _capture_results(self, ctx):
    self.vars.stdout = ctx.results_out
    self.vars.stderr = ctx.results_err
    self.vars.cmd = ctx.cmd
    if self.verbosity >= 4:
        self.vars.run_info = ctx.run_info