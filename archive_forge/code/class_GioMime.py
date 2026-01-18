from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
from ansible_collections.community.general.plugins.module_utils.gio_mime import gio_mime_runner, gio_mime_get
class GioMime(ModuleHelper):
    output_params = ['handler']
    module = dict(argument_spec=dict(mime_type=dict(type='str', required=True), handler=dict(type='str', required=True)), supports_check_mode=True)

    def __init_module__(self):
        self.runner = gio_mime_runner(self.module, check_rc=True)
        self.vars.set_meta('handler', initial_value=gio_mime_get(self.runner, self.vars.mime_type), diff=True, change=True)

    def __run__(self):
        check_mode_return = (0, 'Module executed in check mode', '')
        if self.vars.has_changed('handler'):
            with self.runner.context(args_order=['mime_type', 'handler'], check_mode_skip=True, check_mode_return=check_mode_return) as ctx:
                rc, out, err = ctx.run()
                self.vars.stdout = out
                self.vars.stderr = err
                if self.verbosity >= 4:
                    self.vars.run_info = ctx.run_info