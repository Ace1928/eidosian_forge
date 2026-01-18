from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, Set
from ansible.module_utils.common.collections import is_sequence
from ansible.utils.context_objects import CLIArgs, GlobalCLIArgs
def _init_global_context(cli_args):
    """Initialize the global context objects"""
    global CLIARGS
    CLIARGS = GlobalCLIArgs.from_options(cli_args)