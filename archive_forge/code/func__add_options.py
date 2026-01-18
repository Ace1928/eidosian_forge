from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
@staticmethod
def _add_options(command, env, get_option_value, options):
    if get_option_value is None:
        return
    for option, f in options.items():
        v = get_option_value(option)
        if v is not None:
            f(v, command, env)