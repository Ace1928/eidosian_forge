from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.xfconf import xfconf_runner
class XFConfProperty(StateModuleHelper):
    change_params = ('value',)
    diff_params = ('value',)
    output_params = ('property', 'channel', 'value')
    module = dict(argument_spec=dict(state=dict(type='str', choices=('present', 'absent'), default='present'), channel=dict(type='str', required=True), property=dict(type='str', required=True), value_type=dict(type='list', elements='str', choices=('string', 'int', 'double', 'bool', 'uint', 'uchar', 'char', 'uint64', 'int64', 'float')), value=dict(type='list', elements='raw'), force_array=dict(type='bool', default=False, aliases=['array'])), required_if=[('state', 'present', ['value', 'value_type'])], required_together=[('value', 'value_type')], supports_check_mode=True)
    default_state = 'present'

    def __init_module__(self):
        self.runner = xfconf_runner(self.module)
        self.does_not = 'Property "{0}" does not exist on channel "{1}".'.format(self.vars.property, self.vars.channel)
        self.vars.set('previous_value', self._get())
        self.vars.set('type', self.vars.value_type)
        self.vars.meta('value').set(initial_value=self.vars.previous_value)

    def process_command_output(self, rc, out, err):
        if err.rstrip() == self.does_not:
            return None
        if rc or len(err):
            self.do_raise('xfconf-query failed with error (rc={0}): {1}'.format(rc, err))
        result = out.rstrip()
        if 'Value is an array with' in result:
            result = result.split('\n')
            result.pop(0)
            result.pop(0)
        return result

    def _get(self):
        with self.runner('channel property', output_process=self.process_command_output) as ctx:
            return ctx.run()

    def state_absent(self):
        with self.runner('channel property reset', check_mode_skip=True) as ctx:
            ctx.run(reset=True)
            self.vars.stdout = ctx.results_out
            self.vars.stderr = ctx.results_err
            self.vars.cmd = ctx.cmd
            if self.verbosity >= 4:
                self.vars.run_info = ctx.run_info
        self.vars.value = None

    def state_present(self):
        self.vars.value = [str(v) for v in self.vars.value]
        value_type = self.vars.value_type
        values_len = len(self.vars.value)
        types_len = len(value_type)
        if types_len == 1:
            value_type = value_type * values_len
        elif types_len != values_len:
            self.do_raise('Number of elements in "value" and "value_type" must be the same')
        self.vars.is_array = bool(self.vars.force_array) or isinstance(self.vars.previous_value, list) or values_len > 1
        with self.runner('channel property create force_array values_and_types', check_mode_skip=True) as ctx:
            ctx.run(create=True, force_array=self.vars.is_array, values_and_types=(self.vars.value, value_type))
            self.vars.stdout = ctx.results_out
            self.vars.stderr = ctx.results_err
            self.vars.cmd = ctx.cmd
            if self.verbosity >= 4:
                self.vars.run_info = ctx.run_info
        if not self.vars.is_array:
            self.vars.value = self.vars.value[0]
            self.vars.type = value_type[0]
        else:
            self.vars.type = value_type