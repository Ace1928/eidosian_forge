from __future__ import absolute_import, division, print_function
import shlex
import pipes
import re
import json
import os
def add_arg_to_cmd(cmd_list, param_name, param_value, is_bool=False, omit=None):
    """
    @cmd_list - List of cmd args.
    @param_name - Param name / flag.
    @param_value - Value of the parameter.
    @is_bool - Flag is a boolean and has no value.
    @omit - List of parameter to omit from the command line.
    """
    if param_name.replace('-', '') not in omit:
        if is_bool is False and param_value is not None:
            cmd_list.append(param_name)
            if param_name == '--eval':
                cmd_list.append('{0}'.format(escape_param(param_value)))
            else:
                cmd_list.append(param_value)
        elif is_bool is True:
            cmd_list.append(param_name)
    return cmd_list