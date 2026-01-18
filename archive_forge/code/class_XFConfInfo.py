from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
from ansible_collections.community.general.plugins.module_utils.xfconf import xfconf_runner
class XFConfInfo(ModuleHelper):
    module = dict(argument_spec=dict(channel=dict(type='str'), property=dict(type='str')), required_by=dict(property=['channel']), supports_check_mode=True)

    def __init_module__(self):
        self.runner = xfconf_runner(self.module, check_rc=True)
        self.vars.set('list_arg', False, output=False)
        self.vars.set('is_array', False)

    def process_command_output(self, rc, out, err):
        result = out.rstrip()
        if 'Value is an array with' in result:
            result = result.split('\n')
            result.pop(0)
            result.pop(0)
            self.vars.is_array = True
        return result

    def _process_list_properties(self, rc, out, err):
        return out.splitlines()

    def _process_list_channels(self, rc, out, err):
        lines = out.splitlines()
        lines.pop(0)
        lines = [s.lstrip() for s in lines]
        return lines

    def __run__(self):
        self.vars.list_arg = not (bool(self.vars.channel) and bool(self.vars.property))
        output = 'value'
        proc = self.process_command_output
        if self.vars.channel is None:
            output = 'channels'
            proc = self._process_list_channels
        elif self.vars.property is None:
            output = 'properties'
            proc = self._process_list_properties
        with self.runner.context('list_arg channel property', output_process=proc) as ctx:
            result = ctx.run(**self.vars)
        if not self.vars.list_arg and self.vars.is_array:
            output = 'value_array'
        self.vars.set(output, result)