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
def _is_venv_command(command):
    venv_parser = argparse.ArgumentParser()
    venv_parser.add_argument('-m', type=str)
    argv = shlex.split(command)
    if argv[0] == 'pyvenv':
        return True
    args, dummy = venv_parser.parse_known_args(argv[1:])
    if args.m == 'venv':
        return True
    return False