from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
def _create_repeated(argument_name):

    def f(value, arguments, env):
        for v in value:
            arguments.extend([argument_name, to_native(v)])
    return f