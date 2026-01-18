from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
from ansible_collections.community.general.plugins.module_utils.gconftool2 import gconftool2_runner
class GConftoolInfo(ModuleHelper):
    output_params = ['key']
    module = dict(argument_spec=dict(key=dict(type='str', required=True, no_log=False)), supports_check_mode=True)

    def __init_module__(self):
        self.runner = gconftool2_runner(self.module, check_rc=True)

    def __run__(self):
        with self.runner.context(args_order=['state', 'key']) as ctx:
            rc, out, err = ctx.run(state='get')
            self.vars.value = None if err and (not out) else out.rstrip()