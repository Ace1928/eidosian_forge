from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _get_aliases(self):

    def process(rc, out, err):
        if err:
            return {}
        aliases = [self._RE_ALIAS_LIST.match(a.strip()) for a in out.splitlines()[1:]]
        snap_alias_list = [(entry.group('snap'), entry.group('alias')) for entry in aliases]
        results = {}
        for snap, alias in snap_alias_list:
            results[snap] = results.get(snap, []) + [alias]
        return results
    with self.runner('state_alias name', check_rc=True, output_process=process) as ctx:
        aliases = ctx.run(state_alias='info')
        if self.verbosity >= 4:
            self.vars.get_aliases_run_info = ctx.run_info
        return aliases