from __future__ import absolute_import, division, print_function
import argparse
import os
import re
import sys
import tempfile
import operator
import shlex
import traceback
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, is_executable, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def _recover_package_name(names):
    """Recover package names as list from user's raw input.

    :input: a mixed and invalid list of names or version specifiers
    :return: a list of valid package name

    eg.
    input: ['django>1.11.1', '<1.11.3', 'ipaddress', 'simpleproject>1.1.0', '<2.0.0']
    return: ['django>1.11.1,<1.11.3', 'ipaddress', 'simpleproject>1.1.0,<2.0.0']

    input: ['django>1.11.1,<1.11.3,ipaddress', 'simpleproject>1.1.0,<2.0.0']
    return: ['django>1.11.1,<1.11.3', 'ipaddress', 'simpleproject>1.1.0,<2.0.0']
    """
    tmp = []
    for one_line in names:
        tmp.extend(one_line.split(','))
    names = tmp
    name_parts = []
    package_names = []
    in_brackets = False
    for name in names:
        if _is_package_name(name) and (not in_brackets):
            if name_parts:
                package_names.append(','.join(name_parts))
            name_parts = []
        if '[' in name:
            in_brackets = True
        if in_brackets and ']' in name:
            in_brackets = False
        name_parts.append(name)
    package_names.append(','.join(name_parts))
    return package_names