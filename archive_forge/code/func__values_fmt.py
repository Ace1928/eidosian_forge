from __future__ import absolute_import, division, print_function
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
@fmt.unpack_args
def _values_fmt(values, value_types):
    result = []
    for value, value_type in zip(values, value_types):
        if value_type == 'bool':
            value = 'true' if boolean(value) else 'false'
        result.extend(['--type', '{0}'.format(value_type), '--set', '{0}'.format(value)])
    return result