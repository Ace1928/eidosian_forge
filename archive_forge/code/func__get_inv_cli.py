from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import datetime
import os
import platform
import random
import shlex
import shutil
import socket
import sys
import time
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.loader import module_loader
from ansible.utils.cmd_functions import run_cmd
from ansible.utils.display import Display
@staticmethod
def _get_inv_cli():
    inv_opts = ''
    if context.CLIARGS.get('inventory', False):
        for inv in context.CLIARGS['inventory']:
            if isinstance(inv, list):
                inv_opts += " -i '%s' " % ','.join(inv)
            elif ',' in inv or os.path.exists(inv):
                inv_opts += ' -i %s ' % inv
    return inv_opts