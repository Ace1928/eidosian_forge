from __future__ import absolute_import, division, print_function
import json
import re
from functools import wraps
from itertools import chain
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import CliconfBase
def _get_command_with_output(self, command, output):
    options_values = self.get_option_values()
    if output not in options_values['output']:
        raise ValueError("'output' value %s is invalid. Valid values are %s" % (output, ','.join(options_values['output'])))
    if output == 'json' and (not command.endswith('| display json')):
        cmd = '%s | display json' % command
    elif output == 'xml' and (not command.endswith('| display xml')):
        cmd = '%s | display xml' % command
    elif output == 'text' and (command.endswith('| display json') or command.endswith('| display xml')):
        cmd = command.rsplit('|', 1)[0]
    else:
        cmd = command
    return cmd